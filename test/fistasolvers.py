# This is an old implementation of the FISTA solver

import numba
import numpy as np
import time

@numba.jit('double[:](double[:],double)')
def _soft_threshold_scalar(v, th):
    ret = np.empty_like(v)
    for i in range(ret.size):
        if v[i] > th:
            ret[i] = v[i] - th
        elif v[i] < -th:
            ret[i] = v[i] + th
        else:
            ret[i] = 0
    return ret

@numba.jit('double[:](double[:],double[:])')
def _soft_threshold_vector(v, th):
    ret = np.empty_like(v)
    for i in range(ret.size):
        if v[i] > th[i]:
            ret[i] = v[i] - th[i]
        elif v[i] < -th[i]:
            ret[i] = v[i] + th[i]
        else:
            ret[i] = 0
    return ret

def soft_threshold(v,th):
    if np.isscalar(th):
        return _soft_threshold_scalar(v,th)
    else:
        return _soft_threshold_vector(v,th)


class computationWarp:
    """A wrapper class that provides forward computations with different column/row normalization.

    compAx: a callable that takes a 1D array (size N) as argument and return a 1D array (size M)
    compATy: a callable that takes a 1D array (size M) as arguement and return a 1D array (size N)
    shapeA: (optional) the tuple (M,N)
    """
    def __init__(self, compAx, compATy, shapeA=None):
        if shapeA is not None:
            self._sizeY = shapeA[0]
            self._sizeX = shapeA[1]
        else:
            self._sizeY = None
            self._sizeX = None
        self._compAx = compAx   # original computation
        self._compATy = compATy # original computation
        
        self.compAx = None
        self.compATy = None
        self._Axs = None
        self._ATys = None
        self.resetComps()


    def resetComps(self):
        self.compAx = self._compAx
        self.compATy = self._compATy
        self._Axs = []
        self._ATys = []

    def dims(self):
        return (self._sizeY, self._sizeX)

    def updateColumnWeight(self, colWeight, nocopy=False):
        if self._sizeX is not None:
            # check the size compatibility
            assert self._sizeX == colWeight.size
        
        Ax_RefHolder = self.compAx
        ATy_RefHolder = self.compATy
        if nocopy:
            colWeight = colWeight.ravel()
        else:
            colWeight = colWeight.flatten()
        self._Axs.append(Ax_RefHolder)
        self._ATys.append(ATy_RefHolder)

        def Ax(x):
            return Ax_RefHolder(x*colWeight)
        self.compAx = Ax

        def ATy(y):
            return ATy_RefHolder(y)*colWeight
        self.compATy = ATy


    def updateColumnNorm(self, colNorm, nocopy=False):
        print('updateColumnNorm is deprecated')
        return self.updateColumnWeight(1/colNorm, nocopy)


    def updateRowWeight(self, rowWeight, nocopy=False):
        if self._sizeY is not None:
            # check the size compatibility
            assert self._sizeY == rowWeight.size
        
        Ax_RefHolder = self.compAx
        ATy_RefHolder = self.compATy
        if nocopy:
            rowWeight = rowWeight.ravel()
        else:
            rowWeight = rowWeight.flatten()
        self._Axs.append(Ax_RefHolder)
        self._ATys.append(ATy_RefHolder)

        def Ax(x):
            return Ax_RefHolder(x)*rowWeight
        self.compAx = Ax

        def ATy(y):
            return ATy_RefHolder(y*rowWeight)
        self.compATy = ATy


    def updateOutputROI(self, roi_i_slice, roi_j_slice, outputShape):
        assert np.prod(outputShape) == self._sizeY
        Ax_RefHolder = self.compAx
        ATy_RefHolder = self.compATy
        self._Axs.append(Ax_RefHolder)
        self._ATys.append(ATy_RefHolder)

        def Ax(x):
            Y = Ax_RefHolder(x)
            Y.shape = outputShape
            Y[:, :roi_i_slice.start, :] = 0
            Y[:,  roi_i_slice.stop:, :] = 0
            Y[:,  roi_i_slice,       :roi_j_slice.start] = 0
            Y[:,  roi_i_slice,        roi_j_slice.stop:] = 0
            return Y.ravel()
        self.compAx = Ax

        def ATy(y):
            y = y.copy().reshape(outputShape)
            y[:, :roi_i_slice.start, :] = 0
            y[:,  roi_i_slice.stop:, :] = 0
            y[:,  roi_i_slice,       :roi_j_slice.start] = 0
            y[:,  roi_i_slice,        roi_j_slice.stop:] = 0
            return ATy_RefHolder(y.ravel())
        self.compATy = ATy





class computationWarp2:
    """A wrapper class that provides forward computations with different column/row normalization.
    The input is 

    compAx_core: a callable that takes nothing as argument and return None. Main task is to fft(arrayX), multiply with kernel, ifft(arrayY)
    compATy_core: a callable that takes nothing as argument and return None. Main task is to fft(arrayY), multiply with kernel, ifft(arrayX)
    arrayX: the array for 3D fluorophores
    arrayY: the array for camera images or their related images
    """
    def __init__(self, compAx_core, compATy_core, arrayX, arrayY):
        self._compAx_core = compAx_core   # original computation
        self._compATy_core = compATy_core # original computation
        self._arrayX = arrayX
        self._arrayY = arrayY
        
        self._roiY = None
        self._roiX = None
        self._rowWeight = None
        self._colWeight = None
        self._inputShapeX = None
        self._inputShapeY = None
        self.resetComps()

    def compAx(self, X):
        self._importX(X)
        self._compAx_core() # core function need to clean its output array before dumping
        Y = self._exportArrayY()
        return Y.ravel()


    def compATy(self, Y):
        self._importY(Y)
        self._compATy_core() # core function need to clean its output array before dumping
        X = self._exportArrayX()
        return X.ravel()


    def resetComps(self):
        self._roiY = {'i':slice(0, self._arrayY.shape[1]//2), 'j':slice(0, self._arrayY.shape[2]//2)}
        self._roiX = {'i':slice(0, self._arrayX.shape[1]//2), 'j':slice(0, self._arrayX.shape[2]//2)}
        self._inputShapeY = (self._arrayY.shape[0], self._roiY['i'].stop - self._roiY['i'].start, self._roiY['j'].stop - self._roiY['j'].start)
        self._inputShapeX = (self._arrayX.shape[0], self._roiX['i'].stop - self._roiX['i'].start, self._roiX['j'].stop - self._roiX['j'].start)
        self._rowWeight = None
        self._colWeight = None


    def dims(self):
        print('The shape for Y is', self._inputShapeY, 'and the shape for X is', self._inputShapeX)
        print('The return of dims() is np.prod(shapeY) and np.prod(shapeX)')
        return (np.prod(self._inputShapeY), np.prod(self._inputShapeX))


    def updateColWeightAndRoiX(self, colWeight=None, roi_i=None, roi_j=None, nocopy=False):
        # validating
        if (roi_i is None) and (roi_j is None):
            colweightSize = np.prod(self._inputShapeX)
        elif (roi_i is not None) and (roi_j is not None):
            assert self._validateRoi(roi_i, self._arrayX.shape[1])
            assert self._validateRoi(roi_j, self._arrayX.shape[2])
            newRoiX = {'i':roi_i, 'j':roi_j}
            newInShapeX = (self._arrayX.shape[0], newRoiX['i'].stop - newRoiX['i'].start, newRoiX['j'].stop - newRoiX['j'].start)
            colweightSize = np.prod(newInShapeX)
        else:
            assert 0
        if colWeight is not None:
            assert colWeight.size == colweightSize

        # assigning
        if (roi_i is not None) and (roi_j is not None):
            self._roiX = newRoiX
            self._inputShapeX = newInShapeX
            if self._colWeight is not None and colWeight is None:
                print('updateColWeightAndRoiX: column weight will be reset due to change of ROI and no specified new colWeight')
        
        if colWeight is not None:
            if nocopy:
                self._colWeight = colWeight.reshape(self._inputShapeX)
            else:
                self._colWeight = colWeight.reshape(self._inputShapeX).copy()
        return True


    def updateRowWeightAndRoiY(self, rowWeight=None, roi_i=None, roi_j=None, nocopy=False):
        # validating
        if (roi_i is None) and (roi_j is None):
            rowweightSize = np.prod(self._inputShapeY)
        elif (roi_i is not None) and (roi_j is not None):
            assert self._validateRoi(roi_i, self._arrayY.shape[1])
            assert self._validateRoi(roi_j, self._arrayY.shape[2])
            newRoiY = {'i':roi_i, 'j':roi_j}
            newInShapeY = (self._arrayY.shape[0], newRoiY['i'].stop - newRoiY['i'].start, newRoiY['j'].stop - newRoiY['j'].start)
            rowweightSize = np.prod(newInShapeY)
        else:
            assert 0
        if rowWeight is not None:
            assert rowWeight.size == rowweightSize

        # assigning
        if (roi_i is not None) and (roi_j is not None):
            self._roiY = newRoiY
            self._inputShapeY = newInShapeY
            if self._rowWeight is not None and rowWeight is None:
                print('updateRowWeightAndRoiY: row weight will be reset due to change of ROI and no specified new rowWeight')
        
        if rowWeight is not None:
            if nocopy:
                self._rowWeight = rowWeight.reshape(self._inputShapeY)
            else:
                self._rowWeight = rowWeight.reshape(self._inputShapeY).copy()
        return True


    def _validateRoi(self, roiSlice, M):
        """The roi should be within 0 to M"""
        if roiSlice.start is None or roiSlice.stop is None:
            print('_validateRoi: Using None or negative number is forbidden to avoid ambiguity')
            return False
        if roiSlice.start < 0 or roiSlice.stop < 0:
            print('_validateRoi: Using None or negative number is forbidden to avoid ambiguity')
            return False
        if roiSlice.start >= M or roiSlice.start >= roiSlice.stop:
            print('_validateRoi: empty slice')
            return False
        if roiSlice.stop > M:
            print('_validateRoi: slice stop exceed the range')
            return False
        return True


    def _importX(self,X):
        self._arrayX[:] = 0
        if self._colWeight is None:
            self._arrayX[:,self._roiX['i'],self._roiX['j']] = X.reshape(self._inputShapeX)
        else:
            self._arrayX[:,self._roiX['i'],self._roiX['j']] = X.reshape(self._inputShapeX) * self._colWeight


    def _importY(self,Y):
        self._arrayY[:] = 0
        if self._rowWeight is None:
            self._arrayY[:,self._roiY['i'],self._roiY['j']] = Y.reshape(self._inputShapeY)
        else:
            self._arrayY[:,self._roiY['i'],self._roiY['j']] = Y.reshape(self._inputShapeY) * self._rowWeight


    def _exportArrayX(self):
        if self._colWeight is None:
            return self._arrayX[:,self._roiX['i'],self._roiX['j']].real.copy()
        else:
            return self._arrayX[:,self._roiX['i'],self._roiX['j']].real * self._colWeight


    def _exportArrayY(self):
        if self._rowWeight is None:
            return self._arrayY[:,self._roiY['i'],self._roiY['j']].real.copy()
        else:
            return self._arrayY[:,self._roiY['i'],self._roiY['j']].real * self._rowWeight



# The following fista has a random step to increase the step size
def FISTA( compWrapObj, data, l1_mu, x_init=None, maxiter=10,
           verbose=True, MAX_STEPOUT=1000, L0= 10.0, eta=1.5, progTol =1e-6, posConst=True, L=None,
           decProb=0.1, decEta=2.0,
           objectiveType='LS'):
    """
    This FISTA computation is based on Eric Jonas' implementation

    The L1 norm of columns of A are normalized.
    The normalization is applied by modifying the A matrix. Therefore the intensity 
    is a product of x_opt and the norms of A's columns.
    """

    D, N = compWrapObj.dims()

    if x_init is None:
        x_init = np.zeros(N, dtype=np.float64)

    objective_log = np.zeros(maxiter, dtype=np.float64)
    time_log = np.zeros(maxiter, dtype=np.float64)
    l0_log = np.zeros(maxiter, dtype=int)

    x = x_init.ravel()
    y = x.copy()
    At_yobs = compWrapObj.compATy(data)

    # Computation functions
    if objectiveType == 'LS':
        def f(X,AX=None):
            if AX is not None:
                return 0.5 * np.linalg.norm(AX - data)**2
            else:
                return 0.5 * np.linalg.norm(compWrapObj.compAx(X) - data)**2

        def grad_f(X,AX=None):
            if AX is not None:
                return compWrapObj.compATy(AX) - At_yobs
            else:
                return compWrapObj.compATy(compWrapObj.compAx(X)) - At_yobs
    elif objectiveType == 'Csiszar': # I-divergence
        raise NotImplementedError()
        # this function f is adopted from Bertero 2005,
        # "A simple method for the reduction of boundary effects in the Richardson-Lucy approach to image deconvolution"
        def f(X, AX=None):
            pass

        def grad_f(X,AX=None):
            pass


    def g(X):
        return (l1_mu*np.abs(X)).sum()

    def objective(X, AX=None):
        return f(X,AX) + g(X)

    def proj_L(L, x1, grad_f_x1_cache=None):
        """ 
        argmin (L/2)|| x - [z-(1/L)*grad(z)] ||_2^2 + l1_mu*||x||_1      (x1=z)
          x
        """
        grad_f_x1 = grad_f_x1_cache if grad_f_x1_cache is not None else grad_f(x1)
        x_next = soft_threshold(x1 -  1./L* grad_f_x1, l1_mu/L)

        if posConst:
            x_next[x_next < 0] = 0
        return x_next

    # def F(x):
    #     return objective(x)

    def QL(L, x1, x2, f_x2_cache=None, grad_f_x2_cache=None):
        """
        QL(x1) = f(x2) + grad(x2) . (x1-x2) + (L/2) ||x1-x2||_2^2 + g(x1)
        """
        f_x2 = f_x2_cache if f_x2_cache is not None else f(x2)
        grad_f_x2 = grad_f_x2_cache if grad_f_x2_cache is not None else grad_f(x2)

        return f_x2 + np.dot(x1 - x2, grad_f_x2) + L/2.0 * np.linalg.norm(x1 - x2)**2 + g(x1)


    # Start solving
    if verbose:
        print("k=start", "F(x)=%.11e"%objective(x), "g(x)=%.11e"%g(x) )

    t_stepsize = 1.0
    if L == None:
        # use stepout procedure
        L = L0
        backtracking = True
    else:
        backtracking = False

    time_accum = 0.0
    k=0
    for k in range(maxiter):
        t1 = time.time()
        if verbose:
            print("iter", k, "="*30, flush=True)

        # stochastically increase the step size
        if backtracking:
            if np.random.rand() < decProb:
                print('stochastic increasing the stepsize happens')
                L /= decEta

        # caches for greatly time saving
        AX_for_y = compWrapObj.compAx(y)
        grad_f_y_cache = grad_f(y,AX=AX_for_y)
        # caches for main computation if backtracking is used
        F_y_next_so = None # 'so' means stepout
        y_next_so = None   # 'so' means stepout

        if backtracking:
            f_y_cache = f(y,AX=AX_for_y)
            for i in range(0, MAX_STEPOUT):
                new_L = (eta ** i) * L

                y_next_so = proj_L(L=new_L, x1=y, grad_f_x1_cache=grad_f_y_cache)
                F_y_next_so = objective(y_next_so) # F(y_next_so)
                QL_val = QL(L=new_L, x1=y_next_so, x2=y, f_x2_cache=f_y_cache, grad_f_x2_cache=grad_f_y_cache)

                if verbose:
                    print("Stepping out k=",k, "i=",i, "new_L=",new_L, "F(y_next_so)=%.11e"%F_y_next_so, "QL=%.11e"%QL_val, flush=True)
                if F_y_next_so <= QL_val:
                    break

            #L = (eta ** (i)) * L
            assert new_L == (eta ** (i)) * L
            L = new_L

            #assert L <  1e6
            del f_y_cache

        # FISTA updates, the main body of FISTA
        x_next = y_next_so if y_next_so is not None else proj_L(L=L, x1=y, grad_f_x1_cache=grad_f_y_cache) # the last iteration of backtracking give y_next_so == x_next
        assert np.isnan(x_next).any() == False
        t_stepsize_next = (1 + np.sqrt(1 + 4 * t_stepsize**2))/2.0
        y_next = x_next + ((t_stepsize-1.0)/t_stepsize_next) * (x_next - x)
        assert np.isnan(y_next).any() == False

        x = x_next
        y = y_next
        t_stepsize = t_stepsize_next

        # update logs
        time_accum += (time.time() - t1)
        time_log[k] = time_accum 
        objective_log[k] = F_y_next_so if F_y_next_so is not None else objective(x)
        l0_log[k] = np.sum(x != 0)

        if verbose:
            print("k=%d" % k, "F(x)=%.11e"%objective_log[k], "g(x)=%.11e"%g(x), "L=",L, "t_stepsize=",t_stepsize, flush=True)

        # how much better did we get in percentage terms?
        if k > 1:
            delta_0 = np.abs(objective_log[k] - objective_log[k-1])/objective_log[k-1]
            delta_1 = np.abs(objective_log[k-1] - objective_log[k-2])/objective_log[k-2]
            if delta_0 < progTol and delta_1 < progTol:
                break
    objective_log = objective_log[:k]
    time_log = time_log[:k]
    l0_log = l0_log[:k]

    result_buffer = {'x_est' : x,
            'nIterations' : k+1,
            'objective' : objective_log,
            'times' : time_log, 
            'l0' : l0_log,
            'l1 penalty' : g(x),
            'min_value': objective_log[-1],
            'l1': l1_mu,
            'L':L,
            }

    return x, result_buffer