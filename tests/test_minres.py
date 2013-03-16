import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from MFNG import minres

rng = numpy.random.RandomState(23091)
nparams = 1000

def init_psd_mat(size, damp=0):
    temp = rng.rand(size, size)
    temp += numpy.diag(damp * numpy.ones(size))
    rval = numpy.dot(temp.T, temp)
    return rval / rval.max()

symb = {}
symb['L'] = T.matrix("L")
symb['g'] = T.vector("g")
vals = {}
vals['L'] = init_psd_mat(nparams).astype(floatX)
vals['g'] = rng.rand(nparams).astype(floatX)

## now compute L^-1 g
vals['Linv_g'] = linalg.cho_solve(linalg.cho_factor(vals['L']), vals['g'])

def test_minres():
    rval = minres.minres(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-14,
            damp = 0.,
            maxiter = 10000,
            profile=0)

    f = theano.function([symb['L'], symb['g']], [rval[0][0], rval[1], rval[2]])
    t1 = time.time()
    [Linv_g, flag, iter] = f(vals['L'], vals['g'])
    print 'test_minres runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


def test_minres_xinit():
    symb['xinit'] = T.vector('xinit')
    vals['xinit'] = rng.rand(nparams).astype(floatX)

    symb_Linv_g = minres.minres(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-14,
            damp = 0.,
            maxiter = 10000,
            xinit = [symb['xinit']],
            profile=0)[0]

    f = theano.function([symb['L'], symb['g'], symb['xinit']], symb_Linv_g)
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'], vals['xinit'])[0]
    print 'test_minres_xinit runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


