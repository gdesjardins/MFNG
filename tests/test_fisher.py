import numpy
import time
import theano
import theano.tensor as T
from theano.printing import pydotprint
from scipy import linalg
from MFNG import lincg
from MFNG import fisher
floatX = theano.config.floatX

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2) = (1024,10,11,12)

params = ['W','V','a','b','c']
nparam = N0*N1 + N1*N2 + N0 + N1 + N2

# initialize samples, parameter and x values
vals = {}
vals['v'] = rng.randint(low=0, high=2, size=(M,N0)).astype('float32')
vals['g'] = rng.randint(low=0, high=2, size=(M,N1)).astype('float32')
vals['h'] = rng.randint(low=0, high=2, size=(M,N2)).astype('float32')
vals['W'] = rng.rand(N0, N1).astype('float32')
vals['V'] = rng.rand(N1, N2).astype('float32')
vals['a'] = rng.rand(N0).astype('float32')
vals['b'] = rng.rand(N1).astype('float32')
vals['c'] = rng.rand(N2).astype('float32')
vals['x_W'] = rng.random_sample(vals['W'].shape).astype('float32')
vals['x_V'] = rng.random_sample(vals['V'].shape).astype('float32')
vals['x_a'] = rng.random_sample(vals['a'].shape).astype('float32')
vals['x_b'] = rng.random_sample(vals['b'].shape).astype('float32')
vals['x_c'] = rng.random_sample(vals['c'].shape).astype('float32')
vals['x'] = numpy.hstack((vals['x_W'].flatten(), vals['x_V'].flatten(),
                          vals['x_a'], vals['x_b'], vals['x_c']))

# compute sufficient statistics for each parameter
stats = {}
stats['W'] = vals['v'][:,:,None] * vals['g'][:,None,:]
stats['V'] = vals['g'][:,:,None] * vals['h'][:,None,:]
stats['a'] = vals['v']
stats['b'] = vals['g']
stats['c'] = vals['h']

# gradients are computed as the mean (acrss examples) of sufficient statistics
grads = {}
for param in params:
    grads[param] = stats[param].mean(axis=0)

# Initialize symbols for samples and parameters
symb = {}
for k in ['v','g','h','W','V','x_W','x_V']:
    symb[k] = T.matrix(k)
for k in ['a','b','c','x_a','x_b','x_c']:
    symb[k] = T.vector(k)

### numpy implementation ###
L = numpy.zeros((0,nparam), dtype='float32')
Minv = numpy.float32(1./M)
for i, pi in enumerate(params):
    dim = numpy.prod(vals[pi].shape)
    Li = numpy.zeros((dim,0))
    for j, pj in enumerate(params):
        lterm = (stats[pi] - grads[pi]).reshape(M,-1)
        rterm = (stats[pj] - grads[pj]).reshape(M,-1)
        Lij = Minv * numpy.dot(lterm.T, rterm)
        Li = numpy.hstack((Li, Lij))
    L = numpy.vstack((L, Li))


def test_fisher_diag_implementations():
    Minv = numpy.float32(1./M)
    ### implementation 1 ###
    L1 = []
    for i, pi in enumerate(params):
        lterm = (stats[pi] - grads[pi]).reshape(M,-1)
        rterm = (stats[pi] - grads[pi]).reshape(M,-1)
        L1_pi = Minv * numpy.dot(lterm.T, rterm)
        L1 += [L1_pi]
    ### implementation 2 ###
    samples = [symb['v'], symb['g'], symb['h']]
    symb_L2 = fisher.compute_L_diag(samples)
    f = theano.function(samples, symb_L2)
    L2 = f(vals['v'], vals['g'], vals['h'])
    ### compare both ### 
    for (L1i, L2i) in zip(L1,L2):
        numpy.testing.assert_almost_equal(
                numpy.diag(L1i),
                L2i.flatten(),
                decimal=3)


def test_fisher_Lx():

    Lx = numpy.dot(L, vals['x'])
    Lx_w = Lx[:N0*N1].reshape(N0,N1)
    Lx_v = Lx[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
    Lx_a = Lx[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N0]
    Lx_b = Lx[N0*N1 + N1*N2 + N0 : N0*N1 + N1*N2 + N0 + N1]
    Lx_c = Lx[-N2:]

    ### theano implementation ###
    energies = - T.sum(T.dot(symb['v'], symb['W']) * symb['g'], axis=1) \
               - T.sum(T.dot(symb['g'], symb['V']) * symb['h'], axis=1) \
               - T.dot(symb['v'], symb['a']) \
               - T.dot(symb['g'], symb['b']) \
               - T.dot(symb['h'], symb['c'])

    # test compute_Lx
    symb_params = [symb['W'], symb['V'], symb['a'], symb['b'], symb['c']]
    symb_x = [symb['x_W'], symb['x_V'], symb['x_a'], symb['x_b'], symb['x_c']]
    LLx = fisher.compute_Lx(energies, symb_params, symb_x)

    f_inputs = [symb['v'], symb['g'], symb['h']] + symb_params + symb_x
    f = theano.function(f_inputs, LLx)
    theano.printing.pydotprint(f, outfile='batch_train_func.png')
    
    t1 = time.time()
    rvals = f(vals['v'], vals['g'], vals['h'],
              vals['W'], vals['V'], vals['a'], vals['b'], vals['c'],
              vals['x_W'], vals['x_V'], vals['x_a'], vals['x_b'], vals['x_c'])

    ### compare both implementation ###
    print 'Elapsed: ', time.time() - t1
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

def test_fisher_Linv_x():
    Linv_x = linalg.cho_solve(linalg.cho_factor(L), vals['x'])
    Linv_x_w = Linv_x[:N0*N1].reshape(N0,N1)
    Linv_x_v = Linv_x[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
    Linv_x_a = Linv_x[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N0]
    Linv_x_b = Linv_x[N0*N1 + N1*N2 + N0 : N0*N1 + N1*N2 + N0 + N1]
    Linv_x_c = Linv_x[-N2:]

    energies = -T.sum(T.dot(symb['v'], symb['W']) * symb['g'], axis=1)\
               -T.sum(T.dot(symb['g'], symb['V']) * symb['h'], axis=1)\
               -T.dot(symb['v'], symb['a'])\
               -T.dot(symb['g'], symb['b'])\
               -T.dot(symb['h'], symb['c'])

    def Lx_func(*args):
        symb_params = [symb[p] for p in params]
        Lneg_x = fisher.compute_Lx(energies, symb_params, args)
        return Lneg_x

    [newgrads, niter, rerr] = lincg.linear_cg(
            lambda xw, xv, xa, xb, xc: Lx_func(xw,xv,xa,xb,xc),
            [symb['x_W'], symb['x_V'], symb['x_a'], symb['x_b'], symb['x_c']],
            rtol=1e-10,
            damp = 0.,
            maxiter = 10000)

    f = theano.function([symb['v'], symb['g'], symb['h'],
                         symb['W'], symb['V'], symb['a'], symb['b'], symb['c'],
                         symb['x_W'], symb['x_V'], symb['x_a'], symb['x_b'], symb['x_c']], newgrads)

    [new_dw, new_dv, new_da, new_db, new_dc] = f(
            vals['v'], vals['g'], vals['h'],
            vals['W'], vals['V'], vals['a'], vals['b'], vals['c'],
            vals['x_W'], vals['x_V'], vals['x_a'], vals['x_b'], vals['x_c'])

    numpy.testing.assert_almost_equal(Linv_x_w, new_dw, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_v, new_dv, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_a, new_da, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_b, new_db, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_c, new_dc, decimal=1)


