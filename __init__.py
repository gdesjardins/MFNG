import numpy

import theano

floatX = theano.config.floatX
npy_floatX = getattr(numpy, floatX)
def sharedX(X, name, **kwargs):
    if isinstance(X, numpy.ndarray):
        return theano.shared(numpy.asarray(X, dtype=floatX), name=name, **kwargs)
    elif numpy.isscalar(X):
        return theano.shared(npy_floatX(X), name=name, **kwargs)
    else:
        raise TypeError("Invalid type for function sharedX. Expected ndarray or scalar, not %s" % type(X))


