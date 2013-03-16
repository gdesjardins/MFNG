# Copyright (c) 2013, Guillaume Desjardins, Razvan Pascanu.
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
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..
import theano
import theano.tensor as T
floatX = theano.config.floatX
from theano.printing import Print

def compute_L_diag(samples):
    M = samples[0].shape[0]
    Minv = T.cast(1./M, floatX)
    N = [samples_i.shape[1] for samples_i in samples]

    rval = []
    for (nim1, ni, sim1, si) in zip(N[:-1], N[1:], samples[:-1], samples[1:]):
        Lw_i = Minv * T.dot(sim1.T**2, si**2) - (Minv * T.dot(sim1.T, si))**2
        rval += [Lw_i]
    for (ni, si) in zip(N, samples):
        Lbias_i = T.mean(si**2, axis=0) - T.mean(si, axis=0)**2
        rval += [Lbias_i]

    return rval

def compute_Lx(energies, params, deltas):
    # expectations and derivatives are commutative.
    cenergies = energies - T.mean(energies)
    Minv = T.cast(1./energies.shape[0], floatX)

    rhs_terms = []
    for param_j, delta_j in zip(params, deltas):
        rhs_term = T.Rop(cenergies, param_j, delta_j)
        rhs_terms += [rhs_term]

    Lx_terms = []
    for param_i in params:
        Lx_term = 0
        for rhs in rhs_terms:
            Lx_term += Minv * T.Lop(cenergies, param_i, rhs)
        Lx_terms += [Lx_term]
    return Lx_terms


def compute_Lx_2(grads, params, deltas):
    """
    :param grads: dictionary of (parameters, gradients).
    :param params: list of shared variables (parameters).
    :param deltas: list of T.vector.
    """
    assert len(grads) == len(deltas)
    Lx_term = {}
    for (param, delta) in zip(params, deltas):
        Minv = T.cast(1./param.shape[0], floatX)
        cgrad = grads[param] - T.mean(grads[param], axis=0)
        Lx_term = Minv * T.dot(cgrad.T, T.dot(cgrad, delta))
        Lx_terms += [Lx_term]
    return Lx_terms



