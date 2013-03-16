# Copyright (c) 2013, Razvan Pascanu, Guillaume Desjardins.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the <organization> nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import copy
import theano
import numpy
from theano import tensor
from theano.ifelse import ifelse
from theano.sandbox.scan import scan
from theano.printing import Print
floatX = theano.config.floatX
npy_floatX = getattr(numpy, floatX)


def linear_cg(compute_Ax, b, M=None, xinit = None,
              rtol = 1e-16, maxiter = 100000, damp=0., floatX = None):
    """
    Solves the system A x[i] = b[i], for all i.
    
    When used as part of a Newton-CG method, b is a list of gradients, where each element of
    this list represents a gradient for a given parameter type (i.e. weight or bias of a given
    layer). This method will return a list whose elements approximates A^{-1} b[i], with the
    precision determined by maxiter or the specified tolerance level. This particular
    version implements the Polyak-Ribiere flavor of CG.

    Parameters:
    :param compute_Ax: python function which symbolically computes the matrix-vector product.
    :param b: list of T.vector, corresponding to A x[i] = b[i]
    :param M: list of T.vector (same length as b). Each element is used to precondition its
    corresponding element of the A-diagonal. If [Mi for Mi in M] contains the diagonal elements
    of A, this will implement Jacobi preconditioning.
    :param xinit: list of T.vector (same length as b). x[i] is initial guess for A^{-1} b[i].
    :param rtol: float. CG will stop when the norm of the residual error < rtol.
    :param maxiter: int. Maximum allowable iterations for CG.
    :param damp: float. Damping factor, equivalent to adding a term along the diagonal of A.
    :param floatX: 'float32' or 'float64'.

    Return values:
    rval[0]: niter, number of iterations run by CG
    rval[1]: residual error norm.
    rval[2+i]: approximate value for G^-1 b[i].

    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(b)
    def loop(niter, rkp_norm, *args):
        pk = args[:n_params]
        rk = args[n_params:2*n_params]
        zk = args[2*n_params:3*n_params]
        b  = list(args[3*n_params:4*n_params])
        xk = args[-n_params:]
        A_pk_temp = compute_Ax(*pk)
        A_pk = [A_pk_temp_ + damp*pk_ for A_pk_temp_, pk_ in zip(A_pk_temp, pk)]
        alphak_num = sum((rk_ * zk_).sum() for rk_, zk_ in zip(rk,zk))
        alphak_denum = sum((A_pk_ * pk_).sum() for A_pk_, pk_ in zip(A_pk, pk))
        alphak = alphak_num / alphak_denum
        xkp1 = [xk_ + alphak * pk_ for xk_, pk_ in zip(xk, pk)]
        rkp1 = [rk_ - alphak * A_pk_ for rk_, A_pk_, in zip(rk, A_pk)]
        if M:
            zkp1 = [rkp1_ / m_ for rkp1_, m_ in zip(rkp1, M)]
        else:
            zkp1 = rkp1
        # compute beta_k using Polak-Ribiere
        betak_num = sum((zkp1_* (rkp1_ - rk_)).sum() for rkp1_,rk_,zkp1_ in zip(rkp1,rk,zkp1))
        betak_denum = alphak_num
        betak = betak_num / betak_denum
        pkp1 = [zkp1_ + betak * pk_ for zkp1_, pk_ in zip(zkp1,pk)]
        # compute termination critera
        rkp1_norm = sum((rkp1_**2).sum() for rkp1_ in rkp1)
        b_norm = sum((b_**2).sum() for b_ in b)
        return [niter + 1, rkp1_norm] + pkp1 + rkp1 + zkp1 + b + xkp1,\
               theano.scan_module.until(abs(rkp1_norm) < (rtol * b_norm))
               #theano.scan_module.until(abs(rkp1_norm) < (rtol))

    b0 = [tensor.unbroadcast(tensor.shape_padleft(b_)) for b_ in b]

    # Initialize residual based on xinit
    if xinit is None:
        r0_temp = b
        x0 = [tensor.unbroadcast(tensor.shape_padleft(tensor.zeros_like(b_))) for b_ in b]
    else:
        init_Ax = compute_Ax(*xinit)
        r0_temp = [b[i] - init_Ax[i] for i in xrange(len(b))]
        x0 = [tensor.unbroadcast(tensor.shape_padleft(xinit_)) for xinit_ in xinit]

    # Leftpad r0, z0 and p0 for scan.
    r0 = [tensor.unbroadcast(tensor.shape_padleft(r0_temp_)) for r0_temp_ in r0_temp]
    if M:
        z0 = [tensor.unbroadcast(tensor.shape_padleft(r0_temp_ / m_)) for r0_temp_, m_ in zip(r0_temp, M)]
    else:
        z0 = r0
    p0 = z0

    states = []
    # 0 niter
    states.append(tensor.constant(npy_floatX([0])))
    # 1 residual error norm
    states.append(tensor.constant(npy_floatX([0])))
 
    outs, updates = scan(loop,
                         states = states + p0 + r0 + z0 + b0 + x0,
                         n_steps = maxiter,
                         mode = theano.Mode(linker='c|py'),
                         name = 'linear_conjugate_gradient',
                         profile=0)
    sol = [x[0] for x in outs[-n_params:]]
    niter = outs[0][0]
    rerr = outs[1][0]
    return [sol, niter, rerr]


def linear_cg_fletcher_reeves(compute_Ax, bs, xinit = None,
              rtol = 1e-6, maxiter = 1000, damp=0,
              floatX = None, profile=0):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)


    def loop(rz_old, *args):
        ps = args[:n_params]
        rs = args[n_params:2*n_params]
        xs = args[2*n_params:]
        _Aps = compute_Ax(*ps)
        Aps = [x + damp*y for x,y in zip(_Aps, ps)]
        alpha = rz_old/sum( (x*y).sum() for x,y in zip(Aps, ps))
        xs = [x + alpha * p for x,p in zip(xs,ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        rz_new = sum( (r*r).sum() for r in rs)
        ps = [ r + rz_new/rz_old*p for r,p in zip(rs,ps)]
        return [rz_new]+ps+rs+xs, \
                theano.scan_module.until(abs(rz_new) < rtol)

    if xinit is None:
        r0s = bs
        _x0s = [tensor.unbroadcast(tensor.shape_padleft(tensor.zeros_like(x))) for x in bs]
    else:
        init_Ax = compute_Ax(*xinit)
        r0s = [bs[i] - init_Ax[i] for i in xrange(len(bs))]
        _x0s = [tensor.unbroadcast(tensor.shape_padleft(xi)) for xi in xinit]

    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    rz_old = sum( (r*r).sum() for r in r0s)
    _rz_old = tensor.unbroadcast(tensor.shape_padleft(rz_old),0)
    outs, updates = scan(loop,
                         states = [_rz_old] + _p0s + _r0s + _x0s,
                         n_steps = maxiter,
                         mode = theano.Mode(linker='cvm'),
                         name = 'linear_conjugate_gradient',
                         profile=profile)
    fxs = outs[1+2*n_params:]
    return [x[0] for x in fxs]


