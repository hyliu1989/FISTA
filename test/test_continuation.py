import test.fistasolvers as fistasolvers
import numpy as np
import genericFISTA


class container(object):
    def __init__(self):
        pass


def test_run_0():
    dims = (5,6)

    l1mu = 1.0e-2

    # setup for new fista
    np.random.seed((3,21,4,223))
    A = np.random.rand(5,6)
    b = np.random.rand(5)
    f = lambda x: 0.5*np.linalg.norm(A.dot(x) - b)**2
    gradf = lambda x: A.T.dot(A.dot(x) - b)
    g = lambda x: l1mu*np.linalg.norm(x,1)
    proxg = lambda alpha, x: fistasolvers.soft_threshold(x, alpha*l1mu)

    # run 10
    Logs = genericFISTA.initializeLogs(to_log_all_x=True)
    Logs['objLog'] = []
    fistaObj = genericFISTA.FISTA(f, gradf, g, proxg, gradf_take_cache=False)
    fistaObj.solve(x_init=np.zeros(dims[1], dtype=np.float64), max_iter=10, L0=1.0, ls_red_coef=1.5, ls_inc_mode={'stopAtIter':0},
                   **Logs)

    # run 20 continued
    fistaObj.solve(x_init=np.zeros(dims[1], dtype=np.float64), max_iter=10, L0=1.0, ls_red_coef=1.5, ls_inc_mode={'stopAtIter':0}, prevLastIter=10,
                   **Logs)

    # run 20 direct
    Logs20 = genericFISTA.initializeLogs(to_log_all_x=True)
    Logs20['objLog'] = []
    fistaObj = genericFISTA.FISTA(f, gradf, g, proxg, gradf_take_cache=False)
    fistaObj.solve(x_init=np.zeros(dims[1], dtype=np.float64), max_iter=20, L0=1.0, ls_red_coef=1.5, ls_inc_mode={'stopAtIter':0},
                   **Logs20)

    np.testing.assert_almost_equal( Logs20['xLog'][-1], Logs['xLog'][-1])
    np.testing.assert_almost_equal( Logs20['tLog'],     Logs['tLog'])
    np.testing.assert_almost_equal( Logs20['objLog'],   Logs['objLog'])

    