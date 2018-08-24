# -*- coding: utf-8 -*-
"""
Accelerated proximal gradient descent algorithm with optional restart and callback feature

Example callback are also provided with Plotly.

Hsiou-Yuan Liu   hyhliu1989@gmail.com
Aug 23, 2018

"""
from __future__ import division, print_function, with_statement
import numpy as np
import contexttimer
from os.path import exists
try:
    import arrayfire as af
except:
    af = None

###########
# Utility #
###########
class mylist(list):
    """
    List with limited length which will be check at the time of append()

    This is the list to use if not all the logs are expected to keep
    """
    def __init__(self, noneEmptyLength=10):
        assert noneEmptyLength > 0
        self._n = noneEmptyLength

    def append(self, item):
        super(mylist, self).append(item)
        idx = len(self) - 1 - self._n
        if idx >= 0:
            self[idx] = None


def initializeLogs(to_log_all_x, num_x=0):
    print('deprecated: use static method AcceleratedProximalGD.getInitialLogs() instead')
    return AcceleratedProximalGD.getInitialLogs(to_log_all_x, num_x)


##################
# Main algorithm #
##################
class AcceleratedProximalGD:
    """
    Accelerated proximal gradient descent solver extended from [1] to be able to handle non-l1 
    regularizers.

    reference: 1. Amir Beck and Marc Teboulle
                  A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
                  SIAM J. Imaging Sci., 2(1), 183–202. http://epubs.siam.org/doi/abs/10.1137/080716542
               2. Brendan O’Donoghue and Emmanuel Candes
                  Adaptive Restart for Accelerated Gradient Schemes
                  https://statweb.stanford.edu/~candes/papers/adap_restart_paper.pdf
    """

    @staticmethod
    def getInitialLogs(to_log_all_x, num_x=2):
        """Get a dictionary containing all the logs that the main algorithm can record.

        Default objLog, objFLog, truthCompLog and tLog are deabled. To enable, simply set the key
        to an empty list.

        Users must specify whether to log x for all iterations (to_log_all_x).
        If not, default is to keep the last two x (num_x=2).

        It is recommended to dereference the returned dictionary when calling AcceleratedProximalGD.solve. 
        E.g.
            logs = AcceleratedProximalGD.getInitialLogs()
            solver = AcceleratedProximalGD(f, gradf, g, proxg)
            solver.solve(x_init=something, **logs)
        """
        if not to_log_all_x and num_x <= 0:
            raise ValueError('num_x should be at least 1')
        return dict(objLog=None,         # f(x) + g(x)   times a user specified coefficient (objCoeff)
                    objFLog=None,        # f(x)          times a user specified coefficient (objCoeff)
                    objLog_inexact=[],   # f(y) + g(y)   times a user specified coefficient (objCoeff)
                    objFLog_inexact=[],  # f(y)          times a user specified coefficient (objCoeff)
                    truthCompLog=None,   # |x-x_truth|_2 times a user specified coefficient (truthCompCoeff)
                                         #               the function can also be changed (truthCompMethod)
                    LLog=[],             # Lipschitz constant
                    tLog=[],             # Nesterov acceleration step
                    xLog=[] if to_log_all_x else mylist(num_x),  # current estimate of unknowns
                    timeLog=[]           # wall clock time since execution
                    )


    def __init__(self, f, gradf=None, g=(lambda x: 0.0), proxg=(lambda alpha,x: x), gradf_take_cache=False, f_gradf=None, backend='cpu'):
        """
        Initializer

        f(x): the smooth part of the objective function to be minimized.
              Returns a float number if gradf_take_cache==False, otherwise (float, cache)
        gradf(x) or gradf(x, cache):
              The gradient of f
              Returns an array of the shape of x
        f_gradf: the combined function which returns (f(x), gradf(x)) as a tuple

        g(x): the other part of the objective function, whose proximal operator is cheap to compute
              IMPORTANT NOTE: g itself should contain a regularization parameter. E.g. if using L1 norm to regularize,
                              g(x) = mu*||x||_1
        proxg(alpha, x): the proximal operator of g, which returns argmin alpha*g(y) + 0.5||y-x||_2^2
                                                                      y
                         See important note of g(x). In the example, this proxg computes argmin_y alpha*mu*||x||_1 + 0.5||y-x||_2^2

        """
        assert backend == 'cpu' or backend == 'gpu'
        if gradf is None:
            assert f_gradf is not None
            self._mom_restart = False
        else:
            assert hasattr(f, '__call__') and hasattr(gradf, '__call__')
            self._mom_restart = True
        self._f = f
        self._gradf = gradf
        self._f_gradf = f_gradf
        self._g = g
        self._proxg = proxg
        
        # unify the calling protocol to be always-take-cache
        if not gradf_take_cache:
            self._f = lambda x: (f(x), None)
            self._gradf = lambda x, cache: gradf(x)

        if backend == 'cpu' or af is None:
            if backend == 'gpu' and af is None:
                print('gpu backend is not supported (missing package arrayfire)!')
            self.__norm = np.linalg.norm
            self.__real = np.real
            self.__conj = np.conj
            self.__sum = np.sum
        else:
            self.__norm = af.norm
            self.__real = af.real
            self.__conj = af.conjg
            self.__sum = af.sum


    def _lineSearchProcedure(self, L, y, f_y, gradf_y):
        """Line search procedures specified in FISTA paper"""
        x   = self._proxg(1/L, y-1/L*gradf_y)
        g_x = self._g(x)
        QL  = f_y + self.__sum(self.__real((self.__conj(gradf_y)*(x-y)))) + 0.5*L*self.__norm(x-y)**2 + g_x  # Taylor expansion at y, evaluated at x
        f_x, cache_fx = self._f(x)
        Fx  = f_x + g_x
        passed = Fx <= QL
        return (passed, x, g_x, QL, f_x, Fx, cache_fx)  # usually cache_fx is not used unless we have momentum-restart


    def solve(self, x_init, max_iter=20, verbose=True, callback=None,

              L=None, L0=1.0, ls_red_coef=2.0,
              ls_inc_mode={'stopAtIter':1, 'maxStepOut':2}, ls_inc_coef=0.8,
              ls_to_print=False,

              objLog=None, objLog_inexact=None, objCoeff=1.0,
              objFLog=None, objFLog_inexact=None,
              truthCompLog=None, truth=None, truthCompMethod=(lambda x,tr:np.linalg.norm(x-tr)), truthCompCoeff=1.0,
              LLog=None,
              tLog=None,
              xLog=None,
              timeLog=None,

              flagRestart=False,
              interruptionFilename=None,
              prevLastIter=None,
              ):
        """
        Implementation of Accelerated Proximal Gradient Descent

        Starting from the FISTA paper, I make no assumption that the regularizer should be l1 and
        I reorder the acceleration update (important for correctly logging) a little, such that
        in iteration k, there is no variable of subscript (k+1). The subscripbed value of each
        variable is still the same as that in the FISTA paper. Reordered as follows:
            t[k] = 0.5*( 1+sqrt(1+4*t[k-1]**2) )
            y[k] = x[k-1] + (t[k-1]-1)/t[k] * (x[k-1]-x[k-2]),   if k > 1
                   x[0],   if k==1
            x[k] = prox_L(y[k])
        with initialization t[0] = 0, y[0] = x_init, x[0] = x_init.

        Arguments:
            x_init:   initial point
            max_iter:   maximum number of iterations to run. If continuing previous run (i.e. prevLastIter is used),
                        max_iter means how many extra iterations to run.
            verbose:   control whether to print standard output. This handle is decoupled from stepout print handle
            callback:   callback(x) where x is the estimation at current iteration.
            L:   Lipschitz constant. If not given, line search will be performed.

            [line search related arguments]  (used only when L is None)
            L0:   start point of the search. L is impedence, so a greater L means a smaller step
            ls_red_coef:   L *= ls_red_coef to increase the impedence L
            ls_inc_mode:   default {'stopAtIter':1, 'maxStepOut':2}
                           carry out the increasing-step line search in iterations below ls_inc_mode['stopAtIter'].
                           carry out at most ls_inc_mode['maxStepOut'] for each search.
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
            tLog:   logging t of acceleration iteration if not None
            xLog:   logging x if not None
            timeLog:   logging accumulated elapsed time (wall clock). This includes the time spent in logging
                       process, and it makes huge difference to evaluate extra f(x) when L is given.

            [abortion/continuation]
            interruptionFilename:   if file exists, the iteration stops
            prevLastIter:   if None, current call is a new start.
                            Otherwise it should be an integer > 1, meaning the last iteration before current call.
                            Log should be provided to run the continuation.

            [flags]
            flagRestart:   if True, restart the acceleration (momentum restart) whenever the objective start to go
                           upward. In this case, tLog and objLog_inexact are required.
        """
        ## function call configuration
        L_is_given = (L is not None)
        if not L_is_given:
            assert ls_red_coef > 1  # step reduction should increase L after L *= ls_red_coef
            if ls_inc_mode['stopAtIter'] >= 1:  # carry out at least one step-increase search
                assert ls_inc_coef < 1  # step increasing should reduce L after L *= ls_inc_coef
            L = L0

        if truthCompLog is not None:
            assert truth is not None

        if max_iter == 0:
            raise ValueError('max_iter cannot be zero')

        if not flagRestart:
            if hasattr(self._f_gradf, '__call__'):
                combinedCall = True
                print('Using the combined call for f and gradf')
            else:
                combinedCall = False
        else:  # with restart feature
            assert tLog is not None, 'tLog is required for the restart of APGD.'
            assert objFLog_inexact is not None, 'objFLog_inexact is required for the restart of APGD.'
            assert hasattr(self._f, '__call__')
            assert hasattr(self._gradf, '__call__')


        def Logging():
            to_print = '|'

            if objLog is not None or objFLog is not None:
                if it == 0 or L_is_given:
                    f_x_loc = self._f(x)[0]
                    g_x_loc = self._g(x) if objLog is not None else 0.0
                else:
                    f_x_loc = f_x
                    g_x_loc = g_x
                if objLog is not None:
                    objLog.append(  objCoeff*(f_x_loc+g_x_loc) )
                    to_print += 'obj:%.4e| ' % objLog[-1]
                if objFLog is not None:
                    objFLog.append( objCoeff*(f_x_loc) )
                    to_print += 'smooth obj:%.4e| ' % objFLog[-1]

            if objLog_inexact is not None or objFLog_inexact is not None:
                g_y_loc = self._g(y) if objLog_inexact is not None else 0.0
                if it == 0:
                    f_y_loc = self._f(y)[0]
                else:
                    f_y_loc = f_y
                if objLog_inexact is not None:
                    objLog_inexact.append(  objCoeff*(f_y_loc+g_y_loc) )
                    to_print += 'obj(y):%.4e| ' % objLog_inexact[-1]
                if objFLog_inexact is not None:
                    objFLog_inexact.append( objCoeff*(f_y_loc) )
                    to_print += 'smooth obj(y):%.4e| ' % objFLog_inexact[-1]

            if truthCompLog is not None:
                truthCompLog.append( truthCompCoeff*truthCompMethod(x,truth) )
                to_print += 'truth diff:%.4e| ' % truthCompLog[-1]

            if LLog is not None:
                LLog.append(L)
                to_print += 'L:%.2e| ' % LLog[-1]

            if tLog is not None:
                tLog.append(t)
                to_print += 't:%.2e| ' % tLog[-1]

            if xLog is not None:
                xLog.append(x)

            if timeLog is not None:
                timeLog.append(timer.elapsed)
                to_print += 'time:%.1f| ' % timeLog[-1]

            if verbose:
                print(to_print, flush=True)

            if callback is not None:
                callback(x)


        with contexttimer.Timer() as timer:
            if prevLastIter is None:
                ## new start
                ## zeroth iteration, prepartion work
                it = 0
                if ls_to_print:
                    print('='*15, 'initialize', it, '='*15)
                elif verbose:
                    print('initialize ', end='')
                x = x_init
                y = x_init
                t = 0
                Logging()
                it_beg = 1
            else:
                ## resume from previous run
                ## resuming preparation
                assert prevLastIter > 1
                x_old = xLog[-2]
                x = xLog[-1]
                t = 0
                for _ in range(prevLastIter):
                    t = 0.5*(1+np.sqrt(1+4*t**2))
                if tLog is not None:
                    if tLog[-1] == t:
                        pass  # Good! Things matched
                    else:
                        if not flagRestart:
                            print('previous run might have turned on flagRestart so the t is not a usual APGD t at this iteration.')
                    # assert tLog[-1] == t  # comment out due to incompatibility to the restart feature
                if LLog is not None:
                    L = LLog[-1]
                it_beg = prevLastIter+1
                print('resume previous run with extra max_iter(%d) runs' % max_iter)


            for it in range(it_beg, it_beg + max_iter):
                if ls_to_print:
                    print('='*15, 'iter', it, '='*15)
                elif verbose:
                    print('iter % 4d: ' % it, end='')
                to_print_momentum_restart_happend = False

                #### ACTUAL APGD PART
                x_oldold = 0 if (it == 1) else x_old
                x_old = x
                t_old = t

                t = 0.5*(1+np.sqrt(1+4*t_old**2))
                y = x_old if (it == 1) else x_old + ((t_old-1)/t)*(x_old-x_oldold)


                ### evaluate f(y) and gradf(y), with optional momentum-restart feature
                if not flagRestart:
                    # Flow without momentum-restart
                    #     f_y = f(y)
                    #     gradf_y = gradf(y)
                    if combinedCall:
                        f_y, gradf_y = self._f_gradf(y)
                    else:
                        f_y, cache = self._f(y)
                        gradf_y = self._gradf(y, cache)

                else:  # with momentum-restart feature
                    # Paper for momentum-restart: Adaptive Restart for Accelerated Gradient Schemes
                    # (https://statweb.stanford.edu/~candes/papers/adap_restart_paper.pdf)
                    # Flow with momentum-restart
                    #     f_y = f(y)
                    #     if f_y > f(y_old): # yes, it's f(y_old), not f(x_old), which is different from the paper
                    #          y = x_old
                    #          f_y = f(y)
                    f_y, cache = self._f(y)  # evaluate f(y)

                    # evaluate restart condition
                    if objCoeff*f_y > objFLog_inexact[-1]:  # momentum-restart condition
                        t = 1
                        y = x_old
                        if not L_is_given:  # means that 'f_x' and 'cache_fx_usedInNextIter' are in locals()
                            f_y, cache = f_x, cache_fx_usedInNextIter  # using cache_fx_usedInNextIter from last iteration
                        else:
                            f_y, cache = self._f(y)
                        if verbose:
                            to_print_momentum_restart_happend = True

                    # evaluate gradf(y)
                    gradf_y = self._gradf(y, cache)

                ### Determining L and compute x = prox_L(y)
                if L_is_given:
                    x = self._proxg(1/L, y-1/L*gradf_y)
                else:
                    ## line search
                    # reducing the step size
                    while True:
                        passed, x, g_x, QL, f_x, Fx, cache_fx_usedInNextIter = self._lineSearchProcedure(L, y, f_y, gradf_y)  # assign variables for Logging()
                        if ls_to_print:
                            print('reducing step size:       QL=%.8E,     Fx=%.8E,     L=%.4E' % (QL, Fx, L), flush=True)
                        if passed:
                            break
                        L *= ls_red_coef
                    # increasing the step size
                    if ls_inc_mode['stopAtIter'] >= it:
                        for _ in range(ls_inc_mode['maxStepOut']):
                            L_try = L * ls_inc_coef
                            passed, x_try, g_x_try, QL_try, f_x_try, Fx_try, cache_fx_usedInNextIter = self._lineSearchProcedure(L_try, y, f_y, gradf_y)
                            if ls_to_print:
                                print('increasing step size: QL_try=%.8E, Fx_try=%.8E, L_try=%.4E' % (QL_try, Fx_try, L_try), end='')
                            if passed:
                                L, x, g_x, QL, f_x, Fx = L_try, x_try, g_x_try, QL_try, f_x_try, Fx_try  # assign variables for Logging()
                                if ls_to_print:
                                    print('')
                            else:
                                if ls_to_print:
                                    print(' (break! use previous L)')
                                break

                Logging()
                if to_print_momentum_restart_happend:
                    print('Momentum restart happens at iter %d.' % (it,))
                if interruptionFilename and exists(interruptionFilename):  # external interrupation
                    break

        # return handling
        objval = f_x if 'f_x' in locals() else f_y
        return x, objval

FISTA = AcceleratedProximalGD




try:
    # FIXME: to update the plotly code such that it can be run with the latest plotly and jupyter notebook
    import plotly.graph_objs as go
    import plotly.tools as tls
    from . plotlyStream import JupyterNotebookPlotlyStream

    class CallBack(JupyterNotebookPlotlyStream):
        """A callable class that shows the interating progress at the end of each iteration of FISTA
        through callback.

        The instance of this class is associated with only one dictionary of logs. If another dictionary
        of logs is to use, construct a new CallBack instance.

        logs:   a dictionary containing the logs to show at each iteration
        num_plots_per_row:   number of streaming plots per row
        extras_to_plot:   a list of tuple (name,callable,'scatter'/'heatmap') where callable takes x
                          as the only input and return a real number or a 2D image.
                          If a real number, it will be logged and 'scatter' is anticipated;
                          if a 2D image, it will be displayed and 'heatmap' is anticipated.
        """
        def __init__(self, logs, num_plots_per_row=3, height_of_a_row=240, total_width=900, extras_to_plot=[]):
            super(CallBack, self).__init__()  # super().__init__() # for Python 2 compatibility

            ## initializing inspection
            go_configs = []
            axis_cnt = 0

            ## inspecting logs
            def whetherToLog(log_name): return (log_name in logs.keys() 
                                                and logs[log_name] is not None)
            Log_name_title_pair = [
                ('objLog',          'objective'),
                ('objLog_inexact',  'objective'),
                ('objFLog',         'smooth part of objective'),
                ('objFLog_inexact', 'smooth part of objective'),
                ('truthCompLog',    'diff from truth'),
                ('LLog',            'L (APGD)'),
                ('tLog',            't (APGD'),
                ('timeLog',         'elapsed time'),
            ]
            for log_name, title in Log_name_title_pair:
                if whetherToLog(log_name):
                    axis_cnt += 1
                    if log_name is 'objLog_inexact' and whetherToLog('objLog'):
                        axis_cnt -= 1 # objLog_inexact shares the axis with objLog
                        title = None
                    if log_name is 'objFLog_inexact' and whetherToLog('objFLog'):
                        axis_cnt -= 1 # objFLog_inexact shares the axis with objFLog
                        title = None
                    go_configs.append(
                        dict(title = title,
                             name  = log_name,
                             axis_id = axis_cnt,
                             xaxis = 'x%d' % axis_cnt,
                             yaxis = 'y%d' % axis_cnt,
                             record = logs[log_name],
                             func_x = None,
                             draw_method = go.Scatter,
                             go_id_in_fig = None,
                        )
                    )

            ## inspecting extras_to_plot
            for name, inspection, drawmethod in extras_to_plot:
                assert drawmethod.lower() == 'scatter' or drawmethod.lower() == 'heatmap'
                axis_cnt += 1
                go_configs.append(
                    dict(title = name,
                         name  = name,
                         axis_id = axis_cnt,
                         xaxis = 'x%d' % axis_cnt,
                         yaxis = 'y%d' % axis_cnt,
                         record = [] if drawmethod.lower() == 'scatter' else None,
                         func_x = inspection,
                         draw_method = go.Scatter if drawmethod.lower() == 'scatter' else go.Heatmap,
                         go_id_in_fig = None,
                    )
                )
            if len(go_configs) == 0:
                raise ValueError('Nothing to display')

            ## configure subplot
            n_cols = num_plots_per_row
            n_rows = (axis_cnt-1)//n_cols + 1
            subplot_titles = []
            for conf in go_configs:
                if conf['title'] is not None:  # it is None if the axis is shared
                    subplot_titles.append(conf['title'])
            fig = tls.make_subplots(
                rows=n_rows, 
                cols=n_cols, 
                subplot_titles=subplot_titles
            )
            fig['layout'].update(width=total_width, height=height_of_a_row*n_rows)


            ## fill subplot object
            # go.Scatter objects are good to plot since the contained data 
            #     array will evolve as the iteration increases.
            # go.Heatmap objects requires evaluations of the image at each 
            #     iteration, therefore a dummy z=[[0]]
            for conf in go_configs:
                if conf['draw_method'] is go.Scatter:
                    fig['data'].append(
                        go.Scatter(
                            y=conf['record'], 
                            name=conf['name'],
                            xaxis=conf['xaxis'], 
                            yaxis=conf['yaxis'],
                        )
                    )
                if conf['draw_method'] is go.Heatmap:
                    fig['data'].append(
                        go.Heatmap(
                            z=[[0]], 
                            name=conf['name'],
                            xaxis=conf['xaxis'], 
                            yaxis=conf['yaxis'],
                        )
                    )
                conf['go_id_in_fig'] = len(fig['data'])-1

            ## Update self
            self._go_configs = go_configs
            self._fig = fig

            ## make default to be log scale for objective values
            self.setSublayoutAxis('objLog', yaxis_dict=dict(type='log',autorange=True), debug=False)
            self.setSublayoutAxis('objFLog', yaxis_dict=dict(type='log',autorange=True), debug=False)

        def setSublayout(self, name_or_title):
            pass

        def setSublayoutAxis(self, name_or_title, xaxis_dict=None, yaxis_dict=None, debug=True):
            found = False
            for conf in self._go_configs:
                if conf['title'] == name_or_title or conf['name'] == name_or_title:
                    if found == True:
                        if debug:
                            print('There is are at least two logs with the same name of title')
                    
                    # get the axis id in the subplot setting
                    axis_id = conf['axis_id']

                    # update the axes
                    if xaxis_dict:
                        axis_name = 'xaxis'+str(axis_id)
                        try:
                            self._fig['layout'][axis_name].update(xaxis_dict)
                        except KeyError:
                            self._fig['layout'][axis_name] = {}
                            self._fig['layout'][axis_name].update(xaxis_dict)
                    if yaxis_dict:
                        axis_name = 'yaxis'+str(axis_id)
                        try:
                            self._fig['layout'][axis_name].update(yaxis_dict)
                        except KeyError:
                            self._fig['layout'][axis_name] = {}
                            self._fig['layout'][axis_name].update(yaxis_dict)
                    found = True

            if not found:
                if debug:
                    print('the specified name or title is not found')

        def __call__(self, x):
            """A callback function to be called with AcceleratedProximalGD.solve
            """
            ## Generate heatmap data
            for conf in self._go_configs:
                f = conf['func_x']
                if conf['draw_method'] is go.Heatmap:
                    idx = conf['go_id_in_fig']
                    self._fig['data'][idx].update({'z': f(x)})
                elif f is not None:
                    conf['record'].append(f(x))

            if not self._already_plotted:
                self.firstRun()
                self._already_plotted = True
            else:
                self.update()

except:
    pass


# Old method of streaming the logs
def getPlotlyCallback(Logs_subset, x_inspections=[], charts_per_row=2):
    """
    Return an example callback function that will update plotly offline plots
    for non-None elements in the dictionary Logs_subset.

    x_inspections: a list of (name, callable) pairs that callables take x as
                   the only argument and the return value of each callable is
                   to make a chart.
    """
    print('This function is deprecated. Use CallBack class instead (requiring Plotly installed).')
    import plotly.offline as py
    import plotly.graph_objs as go
    import plotly.tools as tls
    from IPython.display import clear_output

    assert py.offline.__PLOTLY_OFFLINE_INITIALIZED

    scatter_objs_config = []
    axis_cnt = 0

    # count standard charts
    def toLog(log_name): return (log_name in Logs_subset.keys()
                                 and Logs_subset[log_name] is not None)
    Log_name_title_pair = [
        ('objLog',          'objective'),
        ('objLog_inexact',  'objective'),
        ('objFLog',         'smooth part of objective'),
        ('objFLog_inexact', 'smooth part of objective'),
        ('truthCompLog',    'diff from truth'),
        ('LLog',            'L (APGD)'),
        ('tLog',            't (APGD'),
        ('timeLog',         'elapsed time'), 
    ]
    for log_name, title in Log_name_title_pair:
        if toLog(log_name):
            axis_cnt += 1
            if log_name is 'objLog_inexact' and toLog('objLog'):
                axis_cnt -= 1  # plot in objLog's axis
            if log_name is 'objFLog_inexact' and toLog('objFLog'):
                axis_cnt -= 1  # plot in objFLog's axis
            scatter_objs_config.append(
                dict(title  = title,
                     name   = log_name,
                     xaxis  = 'x%d' % axis_cnt,
                     yaxis  = 'y%d' % axis_cnt,
                     record = Logs_subset[log_name],
                     record_need_update=False,
                     record_update_callable=None,
                )
            )
    # count user specified charts
    for name, inspection in x_inspections:
        axis_cnt += 1
        scatter_objs_config.append(
            dict(title  = name,
                 name   = name,
                 xaxis  = 'x%d' % axis_cnt,
                 yaxis  = 'y%d' % axis_cnt,
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
        # remove because they share the same axis
        subplotTitles.remove('objective')
    if subplotTitles.count('smooth part of objective') == 2:
        # remove because they share the same axis
        subplotTitles.remove('smooth part of objective')
    fig = tls.make_subplots(rows=numRows, cols=charts_per_row,
                            subplot_titles=subplotTitles)
    fig['layout'].update(width=900, height=360*numRows)

    # fill subplot object
    for conf in scatter_objs_config:
        fig['data'].append(
            go.Scatter(y=conf['record'], name=conf['name'],
                       xaxis=conf['xaxis'], yaxis=conf['yaxis'],
            )
        )

    def callback(x):
        clear_output(wait=True)
        for i, conf in enumerate(scatter_objs_config):
            if conf['record_need_update']:
                func = conf['record_update_callable']
                conf['record'].append(func(x))
        py.iplot(fig)

    return callback


