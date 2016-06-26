import test.fistasolvers as fistasolvers
import numpy as np
import genericFISTA


class container(object):
    def __init__(self):
        pass


def test_run_0():
    compObj = container()
    compObj.dims = lambda: (5,6)

    l1mu = 1.0e-2

    # setup for old fista
    np.random.seed((3,21,4,223))
    A = np.random.rand(5,6)
    b = np.random.rand(5)
    compObj.compAx = lambda x: A.dot(x)
    compObj.compATy = lambda y: A.T.dot(y)

    # setup for new fista
    f = lambda x: 0.5*np.linalg.norm(A.dot(x) - b)**2
    gradf = lambda x: A.T.dot(A.dot(x) - b)
    g = lambda x: l1mu*np.linalg.norm(x,1)
    proxg = lambda alpha, x: fistasolvers.soft_threshold(x, alpha*l1mu)


    x_opt, resbuf = fistasolvers.FISTA( compObj, b, l1_mu=l1mu, x_init=None, maxiter=10,
           verbose=True, MAX_STEPOUT=1000, L0= 1.0, eta=1.5, progTol =0e-6, posConst=False, L=None,
           decProb=0.0, decEta=2.0,
           objectiveType='LS')


    Logs = genericFISTA.initializeLogs(to_log_all_x=True)
    Logs['objLog'] = []
    fistaObj = genericFISTA.FISTA(f, gradf, g, proxg, gradf_take_cache=False)
    fistaObj.solve(x_init=np.zeros(compObj.dims()[1], dtype=np.float64), max_iter=10, L0=1.0, ls_red_coef=1.5, ls_inc_mode={'stopAtIter':0},
                   **Logs)

    np.testing.assert_almost_equal( x_opt, Logs['xLog'][-1])
    