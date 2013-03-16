import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from theano_optimize.minresQLP import minresQLP

rng = numpy.random.RandomState(23091)
nparams = 1000

def init_psd_mat(size):
    temp = rng.rand(size, size)
    return numpy.dot(temp.T, temp)

vals = {}
vals['L'] = init_psd_mat(nparams).astype(floatX)
vals['g'] = rng.rand(nparams).astype(floatX)

L = theano.shared(vals['L'], name='L')
g = theano.shared(vals['g'], name='g')

## now compute L^-1 g
vals['Linv_g'] = linalg.cho_solve(linalg.cho_factor(vals['L']), vals['g'])

def test_minres():

    sol, flag, iters, relres, Anorm, Acond = minresQLP(
            lambda x: ([T.dot(L, x)], {}),
            g,
            param_shapes = (nparams,),
            rtol=1e-20,
            maxit = 100000)

    f = theano.function([], [sol])
    t1 = time.time()
    rvals = f()
    Linv_g = rvals[0]
    print 'test_minres runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=2)
