import numpy
import time
import copy
import theano
import theano.tensor as T
from scipy import linalg
from scipy.sparse.linalg import cg
from MFNG import lincg
floatX = theano.config.floatX

rng = numpy.random.RandomState(23091)
nparams = 1000

def init_psd_mat(size, damp=0):
    temp = rng.rand(size, size)
    temp += numpy.diag(damp * numpy.ones(size))
    rval = numpy.dot(temp.T, temp)
    return rval / rval.max()

def hilbert(size):
    h = numpy.zeros((size, size))
    for i in xrange(100):
        for j in xrange(100):
            h[i,j] = 1./ (i + j + 1.)
            if i==j: h[i,j] += 0.1
    return h

symb = {}
symb['L'] = T.matrix("L")
symb['g'] = T.vector("g")

vals = {}
vals['L'] = init_psd_mat(nparams, damp=0.).astype(floatX)
vals['g'] = rng.rand(nparams).astype(floatX)

## now compute L^-1 g
vals['Linv_g'] = linalg.cho_solve(linalg.cho_factor(vals['L']), vals['g'])

def test_lincg():
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'])
    print 'test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


def test_lincg_xinit():
    symb['xinit'] = T.vector('xinit')
    vals['xinit'] = rng.rand(nparams).astype(floatX)

    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            xinit = [symb['xinit']],
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['xinit']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'], vals['xinit'])
    print 'test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


def test_lincg_Ldiag_precond():
    symb['M'] = T.vector('M')
    vals['M'] = numpy.diag(vals['L'])
    vals['Ldiag'] = numpy.zeros_like(vals['L'])
    for i in xrange(len(vals['L'])):
        vals['Ldiag'][i,i] = vals['M'][i]
    vals['Ldiag_inv_g'] = linalg.cho_solve(linalg.cho_factor(vals['Ldiag']), vals['g'])

    ### without preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['Ldiag'], vals['g'])
    print 'No precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Ldiag_inv_g'], decimal=3)

    ### with preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = [symb['M']],
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['M']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['Ldiag'], vals['g'], vals['M'])
    print 'With precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Ldiag_inv_g'], decimal=3)

    ### test scipy implementation ###
    t1 = time.time()
    cg(vals['Ldiag'], vals['g'], maxiter=10000, tol=1e-10)
    print 'scipy.sparse.linalg.cg (no preconditioning): Elapsed ', time.time() - t1
    t1 = time.time()
    cg(vals['Ldiag'], vals['g'], maxiter=10000, tol=1e-10, M=numpy.diag(vals['M']))
    print 'scipy.sparse.linalg.cg (preconditioning): Elapsed ', time.time() - t1


def test_lincg_L_diag_heavy_precond():
    vals['Ldh'] = copy.copy(vals['L'])
    rdiag = numpy.random.rand(nparams) * 100
    for i in xrange(len(vals['L'])):
        vals['Ldh'][i,i] += rdiag[i]
    vals['Ldh_inv_g'] = linalg.cho_solve(linalg.cho_factor(vals['Ldh']), vals['g'])
    symb['M'] = T.vector('M')
    vals['M'] = numpy.diag(vals['Ldh'])

    ### without preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            rtol=1e-20,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['Ldh'], vals['g'])
    print 'No precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Ldh_inv_g'], decimal=3)

    ### with preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = [symb['M']],
            rtol=1e-20,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['M']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['Ldh'], vals['g'], vals['M'])
    print 'With precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Ldh_inv_g'], decimal=3)

    ### test scipy implementation ###
    t1 = time.time()
    cg(vals['Ldh'], vals['g'], maxiter=10000, tol=1e-10)
    print 'scipy.sparse.linalg.cg (no preconditioning): Elapsed ', time.time() - t1
    t1 = time.time()
    cg(vals['Ldh'], vals['g'], maxiter=10000, tol=1e-10, M=numpy.diag(1./vals['M']))
    print 'scipy.sparse.linalg.cg (preconditioning): Elapsed ', time.time() - t1


def test_lincg_L_precond():
    symb['M'] = T.vector('M')
    vals['M'] = numpy.diag(vals['L'])

    ### without preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'])
    print 'No precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)

    ### with preconditioning ###
    [sol, niter, rerr] = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = [symb['M']],
            rtol=1e-8,
            maxiter = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['M']], sol + [niter, rerr])
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'], vals['M'])
    print 'With precond: test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


def test_lincg_fletcher():
    rval = lincg.linear_cg_fletcher_reeves(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-20,
            damp = 0.,
            maxiter = 10000,
            floatX = floatX,
            profile=0)

    f = theano.function([symb['L'], symb['g']], rval[0])
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'])
    print 'test_lincg runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)


def test_lincg_fletcher_xinit():
    symb['xinit'] = T.vector('xinit')
    vals['xinit'] = rng.rand(nparams).astype(floatX)

    rval = lincg.linear_cg_fletcher_reeves(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-20,
            damp = 0.,
            maxiter = 10000,
            floatX = floatX,
            xinit = [symb['xinit']],
            profile=0)

    f = theano.function([symb['L'], symb['g'], symb['xinit']], rval[0])
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'], vals['xinit'])
    print 'test_lincg runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=3)

