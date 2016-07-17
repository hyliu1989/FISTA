# -*- coding: utf-8 -*-
"""

generic FISTA handle


Main: class FISTA
Utilities: class mylist, function initializeLogs

Hsiou-Yuan Liu   hyliu@berkeley.edu
Sep 2, 2016 

"""
import numpy as np
import contexttimer
from os.path import exists


###########
# Utility #
###########
class mylist(list):
    """A list to use if not all the logs are expected to keep"""
    def __init__(self, noneEmptyLength=10):
        assert noneEmptyLength > 0
        self._n = noneEmptyLength
    def append(self, item):
        super(mylist, self).append(item)
        idx = len(self) - 1 - self._n
        if idx >= 0:
            self[idx] = None

def initializeLogs(to_log_all_x, num_x=0):
    """
    the returned object can be dereferenced in calling FISTA.solve
    E.g. 
        logs = initializeLogs()
        fista_handle = FISTA(f, gradf, g, proxg)
        fista_handle.solve( x_init=something, **logs )
    """
    if not to_log_all_x and num_x is 0:
        raise ValueError('num_x should be at least 1')
    return dict(objLog=None,
                objFLog=None,
                objLog_inexact=[],
                objFLog_inexact=[],
                truthCompLog=None,
                LLog=[],
                tLog=[],
                xLog=[] if to_log_all_x else mylist(num_x),
                timeLog=[])


def getPlotlyCallback(Logs_subset, x_inspections=[], charts_per_row=2):
    """
    A example callback function that will update the plotly offline plots showing 
    any non-None elements in the dictionary Logs_subset.

    x_inspections: a list of (name, callable) pairs that callables take x as the only argument and
                   the return value of each callable is to make a chart.
    """
    import plotly.offline as py
    import plotly.graph_objs as go
    import plotly.tools as tls
    from IPython.display import clear_output

    assert py.offline.__PLOTLY_OFFLINE_INITIALIZED and tls._ipython_imported

    scatter_objs_config = []
    axis_cnt = 0

    # count standard charts
    toLog = lambda log_name: (log_name in Logs_subset.keys()) and (Logs_subset[log_name] is not None)
    standardLogs =      ['objLog',    'objLog_inexact', 'objFLog',                  'objFLog_inexact']
    standardLogTitles = ['objective', 'objective',      'smooth part of objective', 'smooth part of objective']
    standardLogs +=      ['truthCompLog',    'LLog',       'tLog',      'timeLog'     ]
    standardLogTitles += ['diff from truth', 'L (FISTA)',  't (FISTA)', 'elapsed time']
    for log_name, title in zip(standardLogs,standardLogTitles):
        if toLog(log_name):
            axis_cnt += 1
            if log_name is 'objLog_inexact' and toLog('objLog'):
                axis_cnt -= 1 # plot in objLog's axis
            if log_name is 'objFLog_inexact' and toLog('objFLog'):
                axis_cnt -= 1 # plot in objFLog's axis
            scatter_objs_config.append( dict(
                title = title,
                name  = log_name,
                xaxis = 'x%d' % axis_cnt, 
                yaxis = 'y%d' % axis_cnt,
                record = Logs_subset[log_name],
                record_need_update=False,
                record_update_callable=None,
                )
            )
    # count user specified charts
    for name, inspection in x_inspections:
        axis_cnt += 1
        scatter_objs_config.append(dict(
            title = name,
            name  = name,
            xaxis = 'x%d' % axis_cnt, 
            yaxis = 'y%d' % axis_cnt,
            record = [],
            record_need_update=True,
            record_update_callable=inspection,
            )
        )

    if len(scatter_objs_config) == 0:
        raise ValueError('Nothing to display')

    # generate subplot figure object
    numRows = (axis_cnt-1)//charts_per_row + 1
    subplotTitles = [conf['title'] for conf in scatter_objs_config]
    if subplotTitles.count('objective') == 2:
        subplotTitles.remove('objective') # remove because they share the same axis
    if subplotTitles.count('smooth part of objective') == 2:
        subplotTitles.remove('smooth part of objective') # remove because they share the same axis
    fig = tls.make_subplots(rows=numRows, cols=charts_per_row, subplot_titles=subplotTitles)
    fig['layout'].update(width=900, height=360*numRows)
    
    # fill subplot object
    for conf in scatter_objs_config:
        fig['data'].append(go.Scatter(y=conf['record'], xaxis=conf['xaxis'], yaxis=conf['yaxis'], name=conf['name']))

    def callback(x):
        clear_output(wait=True)
        for i, conf in enumerate(scatter_objs_config):
            if conf['record_need_update']:
                func = conf['record_update_callable']
                conf['record'].append(func(x))
        py.iplot(fig)

    return callback




##################
# Main algorithm #
##################
class FISTA:
    """
    FISTA solver class to handle the procedures in each FISTA iterations
    """
    def __init__(self, f, gradf, g, proxg, gradf_take_cache=False):
        """
        Initializer

        f(x): the smooth part of the objective function to be minimized.
              Returns a float number if gradf_take_cache==False, otherwise (float, cache)
        gradf(x) or gradf(x, cache): 
              The gradient of f
              Returns an array of the shape of x
        g(x): the other part of the objective function, whose proximal operator is cheap to compute
              IMPORTANT NOTE: g itself should contain a regularization parameter. E.g. if using L1 norm to regularize,
                              g(x) = mu*||x||_1
        proxg(alpha, x): the proximal operator of g, which returns argmin alpha*g(y) + 0.5||y-x||_2^2
                                                                      y
                         See important note of g(x). In the example, this proxg computes argmin_y alpha*mu*||x||_1 + 0.5||y-x||_2^2

        """
        assert hasattr(f, '__call__') and hasattr(gradf, '__call__') and hasattr(g, '__call__') and hasattr(proxg, '__call__') 
        self._f = f
        self._gradf = gradf
        self._g = g
        self._proxg = proxg
        self._gradf_take_cache = gradf_take_cache

    def _lineSearchProcedure(self, L, y, f_y, gradf_y):
        """Line search procedures specified in FISTA paper"""
        x   = self._proxg(1/L, y-1/L*gradf_y)
        g_x = self._g(x)
        QL  = f_y + (gradf_y*(x-y)).sum() + 0.5*L*np.linalg.norm(x-y)**2 + g_x # Taylor expansion at y, evaluated at x
        f_x = self._f(x) if not self._gradf_take_cache else self._f(x)[0]
        Fx  = f_x + g_x
        passed = Fx < QL
        return (passed, x, g_x, QL, f_x, Fx)


    def solve(self, x_init, max_iter=20, verbose=True, callback=None,

              L=None, L0=1.0, ls_red_coef=2.0,
              ls_inc_mode={'stopAtIter':1, 'maxStepOut':2}, ls_inc_coef=0.8, ls_to_print=False,

              objLog=None, objLog_inexact=None, objCoeff=1.0,
              objFLog=None, objFLog_inexact=None,
              truthCompLog=None, truth=None, truthCompMethod=lambda x,tr:np.linalg.norm(x-tr), truthCompCoeff=1.0,
              LLog=None,
              tLog=None,
              xLog=None,
              timeLog=None,

              interruptionFilename=None,
              prevLastIter=None
              ):
        """
        Implementation of FISTA

        Here I reorder the FISTA update (important for correctly logging) a little, such that
        in iteration k, there is no variable of subscript (k+1). The subscripbed value of each 
        variable is still the same as that in the FISTA paper. Reordered as follows:
            t[k] = 0.5*( 1+sqrt(1+4*t[k-1]**2) )
            y[k] = x[k-1] + (t[k-1]-1)/t[k] * (x[k-1]-x[k-2]),   if k > 1
                   x[0],   if k==1
            x[k] = prox_L(y[k])
        with initialization t[0] = 0, y[0] = x_init, x[0] = x_init.

        Arguments:
            x_init:   initial point
            max_iter:   maximum iteration for FISTA to run
            verbose:   control whether to print standard output. This handle is decoupled from stepout print handle
            callback:   callback(x) where x is the estimation at current iteration.
            L:   Lipschitz constant. If not given, line search will be performed.

            [line search related arguments]  (used only when L is None)
            L0:   start point of the search. L is impedence, so a greater L means a smaller step
            ls_red_coef:   L *= ls_red_coef to increase the impedence L
            ls_inc_mode:   carry out the increasing-step line search in iterations below ls_inc_mode['stopAtIter']
                           carry out at most ls_inc_mode['maxStepOut'] for each search
            ls_inc_coef:   L *= ls_inc_coef to decrease the impedence L
            ls_to_print:   stepout printing handle

            [log related arguments]  (all the logs should have `.append()` method)
            objLog:   logging the objective if not None
            objFLog:   logging the smooth part of the objective if not None
            objLog_inexact:   using y to evaluate the objective if not None.
            objFLog_inexact:   using y to evaluate the smooth part of the objective if not None
            objCoeff:   value to multiply with the objective for logging. [Default=1.0]
                        Example of use, to normalize the objective if set to (1/norm_value).
            truthCompLog:   logging the difference between x and a given `truth` if not None.
            truth:   the given truth
            truthCompMethod:   a function to compute difference. [ Default=lambda x,tru: np.linalg.norm(x-tru) ]
            truthCompCoeff:   value to multiply with the difference for logging. [Default=1.0]
                               Example of use, to normalize the difference.
            LLog:   logging L if not None
            tLog:   logging t of FISTA iteration if not None
            xLog:   logging x if not None
            timeLog:   logging accumulated elapsed time

            [abortion/continuation]
            interruptionFilename:   if file exists, the iteration stops
            prevLastIter:   if None, current call is a new start.
                            Otherwise it should be an integer > 1, meaning the last iteration before current call.
                            Log should be provided to run the continuation.
        """
        ## function call configuration
        L_is_given = (L is not None)
        if not L_is_given:
            assert ls_red_coef > 1 # step reduction should increase L after L *= ls_red_coef
            if ls_inc_mode['stopAtIter'] >= 1: # carry out at least one step-increase search
                assert ls_inc_coef < 1 # step increasing should reduce L after L *= ls_inc_coef
            L = L0

        if truthCompLog is not None:
            assert truth is not None

        if max_iter == 0:
            raise ValueError('max_iter cannot be zero')

        def Logging():
            to_print = '|'

            if objLog is not None or objFLog is not None:
                if it == 0 or L_is_given:
                    f_x_loc = self._f(x) if not self._gradf_take_cache else self._f(x)[0]
                    g_x_loc = self._g(x) if objLog is not None else 0.0
                else:
                    f_x_loc = f_x
                    g_x_loc = g_x
                if objLog is not None:  
                    objLog.append(  objCoeff*(f_x_loc+g_x_loc) )
                    to_print += 'obj:%.4e| ' % objLog[-1]
                if objFLog is not None: 
                    objFLog.append( objCoeff*(f_x_loc) )
                    to_print += 'smooth obj:%.4e| ' % objLog[-1]

            if objLog_inexact is not None or objFLog_inexact is not None:
                g_y_loc = self._g(y) if objLog_inexact is not None else 0.0
                if it == 0:
                    f_y_loc = self._f(y) if not self._gradf_take_cache else self._f(y)[0]
                else:
                    f_y_loc = f_y
                if objLog_inexact is not None:
                    objLog_inexact.append(  objCoeff*(f_y_loc+g_y_loc) )
                    to_print += 'obj(inexact):%.4e| ' % objLog_inexact[-1]
                if objFLog_inexact is not None:
                    objFLog_inexact.append( objCoeff*(f_y_loc) )
                    to_print += 'smooth obj(inexact):%.4e| ' % objFLog_inexact[-1]

            if truthCompLog is not None:
                truthCompLog.append( truthCompCoeff*truthCompMethod(x,truth) )
                to_print += 'truth diff:%.4e| ' % truthCompLog[-1]

            if LLog is not None:
                LLog.append(L)
                to_print += 'L:%.4e| ' % LLog[-1]

            if tLog is not None:
                tLog.append(t)
                to_print += 't:%.4e| ' % tLog[-1]

            if xLog is not None:
                xLog.append(x)

            if timeLog is not None:
                timeLog.append(0.0 if it==0 else timer.elapsed)
                to_print += 'time:%.1f| ' % timeLog[-1]
            
            if verbose: 
                print(to_print)

            if callback is not None:
                    callback(x)


        if prevLastIter is None:
            # new start
            # zeroth iteration, prepartion work
            if ls_to_print: print('='*15, 'initialize', it, '='*15)
            elif verbose:   print('initialize ', end='')
            it = 0
            x = x_init
            y = x_init
            t = 0
            Logging()
            it_beg = 1
        else:
            # resume from previous run
            # resuming preparation
            assert prevLastIter > 1
            x_old = xLog[-2]
            x = xLog[-1]
            t = 0
            for _ in range(prevLastIter): 
                t = 0.5*(1+np.sqrt(1+4*t**2))
            if tLog is not None:
                assert tLog[-1] == t
            if LLog is not None:
                L = LLog[-1]
            it_beg = prevLastIter+1
            


        with contexttimer.Timer() as timer:
            for it in range(it_beg, it_beg + max_iter):
                if ls_to_print: print('='*15, 'iter', it, '='*15)
                elif verbose:   print('iter % 4d: ' % it, end='')

                #### ACTUAL FISTA PART
                x_oldold = 0 if (it==1) else x_old
                x_old = x
                t_old = t

                t = 0.5*(1+np.sqrt(1+4*t_old**2))
                y = x_old if (it==1) else x_old + ((t_old-1)/t)*(x_old-x_oldold)

                ### evaluate f(y) and gradf(y)
                if not self._gradf_take_cache:
                    f_y = self._f(y)
                    gradf_y = self._gradf(y)
                else:
                    f_y, cache = self._f(y)
                    gradf_y = self._gradf(y,cache)

                ### Determining L and compute x = prox_L(y)
                if L_is_given:
                    x = self._proxg(1/L, y-1/L*gradf_y)
                else:
                    ## line search
                    # reducing the step size
                    while True:
                        passed, x, g_x, QL, f_x, Fx = self._lineSearchProcedure(L, y, f_y, gradf_y) # assign variables for Logging()
                        if ls_to_print:
                            print('reducing step size:       QL=%.8E,     Fx=%.8E,     L=%.4E' %(QL, Fx, L), flush=True)
                        if passed:
                            break
                        L *= ls_red_coef
                    # increasing the step size
                    if ls_inc_mode['stopAtIter'] >= it:
                        for _ in range(ls_inc_mode['maxStepOut']):
                            L_try = L * ls_inc_coef
                            passed, x_try, g_x_try, QL_try, f_x_try, Fx_try = self._lineSearchProcedure(L_try, y, f_y, gradf_y)
                            if ls_to_print:
                                print('increasing step size: QL_try=%.8E, Fx_try=%.8E, L_try=%.4E' %(QL_try, Fx_try, L_try), end='')
                            if passed:
                                L, x, g_x, QL, f_x, Fx = L_try, x_try, g_x_try, QL_try, f_x_try, Fx_try # assign variables for Logging()
                                if ls_to_print: print('')
                            else:
                                if ls_to_print: print(' (break! use previous L)')
                                break
                
                Logging()
                if interruptionFilename and exists(interruptionFilename): # external interrupation
                    break

        # return handling
        objval = f_x if 'f_x' in locals() else f_y
        return x, objval






