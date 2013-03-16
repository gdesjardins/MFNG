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
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This script implements Annealed Importance Sampling for estimating the partition
function of an n-layer DBM, as described in:
On the Quantitative Analysis of Deep Belief Networks, Salakhutdinov & Murray
Deep Boltzmann Machines, Salakhutdinov & Hinton.

Denote H={h1,h2,h3,...,hK} the set of all hidden layers. Further define He to
be the set of of all even-numbered layers, and Ho, the set of all odd-numbered
layers. For notational simplicity, we define h0:=v.

The DBM energy function being considered is the following:
    E(v, H) = -\sum_{k=1}^{K} h_{k-1}^T Wk hk -\sum_{k=0}^K hk^T bk

This implementation works by interpolating between:

* the target distribution p_B(v), with free energy Fe_B(v, He) obtained by
  marginalizing all odd layers.  TODO

* the baserate distribution p_A(h1): TODO

* using the interpolating distributions p_k(h^1) defined as:
  p_k*(h1) \proto p*_A^(1 - \beta_k) p*_B^(beta_k).

To maximize accuracy, the model p_A is chosen to be the maximum likelihood
solution when all weights are null (i.e. W1=W2=0). We refer to b1_A as the
baserate biases. They can be computed directly from the dataset. Note that
because all h1 units of p_A are independent, we can easily compute the partition
function Z_A, as:

    Z_A = 2^N0 2^N2 \prod_j (1 + exp(b1_j))

where N0 and N2 are the number of hidden units at layers 0 and 2.

The ratio of partition functions Z_B / Z_A is then estimated by averaging the
importance weights w^(m), given by:
    
            p*_1(h_0) p*_2(h_1)     p*_K(h_{K-1})
    w^(m) = --------- --------- ... -----------------,  where h_k ~ p_k(h1).
            p*_0(h_0) p*_1(h_1)     p*_{K-1}(h_{K-1})

    log w^(m) = FE_0(h_0) - FE_1(h_0) + FE_1(h_1) - FE_2(h_1) + etc.
"""
import numpy
import logging
import optparse
import time
import pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.utils import serial
from pylearn2.datasets import mnist

floatX = theano.config.floatX
logging.basicConfig(level=logging.INFO)

   
def _sample_even_odd(dbm, samples, beta, odd=True):
    for i in xrange(odd, len(samples), 2):
        samples[i] = dbm.sample_hi_given(samples, i, beta)

def _activation_even_odd(dbm, samples, beta, odd=True):
    for i in xrange(odd, len(samples), 2):
        samples[i] = dbm.hi_given(samples, i, beta, apply_sigmoid=False)

def neg_sampling(dbm, nsamples, beta=1.0, pa_bias=None,
                 marginalize_odd=True, theano_rng=None):
    """
    Generate a sample from the intermediate distribution defined at inverse
    temperature `beta`, starting from state `nsamples`. See file docstring for
    equation of p_k(h1).

    Inputs
    ------
    dbm: dbm.DBM object
        DBM from which to sample.
    nsamples: array-like object of shared variables.
        nsamples[i] contains current samples from i-th layer.
    beta: scalar
        Temperature at which we should sample from the model.

    Returns
    -------
    new_nsamples: array-like object of symbolic matrices
        new_nsamples[i] contains new samples for i-th layer.
    """
    new_nsamples = [nsamples[i] for i in xrange(dbm.depth)]
    ### contribution from model B, at temperature beta_k
    _sample_even_odd(dbm, new_nsamples, beta, odd = marginalize_odd)
    _activation_even_odd(dbm, new_nsamples, beta, odd = not marginalize_odd)
    ### contribution from model A, at temperature (1 - beta_k)
    new_nsamples[not marginalize_odd] += pa_bias * (1. - beta)
    start_idx = not 0 if marginalize_odd else 1
    # loop over all layers (not being marginalized)
    for i in xrange(not marginalize_odd, dbm.depth, 2):
        new_nsamples[i] = T.nnet.sigmoid(new_nsamples[i])
        new_nsamples[i] = theano_rng.binomial(
                            size = (dbm.batch_size, dbm.n_u[i]),
                            n=1, p=new_nsamples[i], dtype=floatX)
    return new_nsamples


def free_energy_at_beta(model, samples, beta, pa_bias=None,
                        marginalize_odd=True):
    """
    Computes the free-energy of the sample `h1_sample`, for model p_k(h1).

    Inputs
    ------
    h1_sample: theano.shared
        Shared variable representing a sample of layer h1.
    beta: T.scalar
        Inverse temperature beta_k of model p_k(h1) at which to measure the free-energy.

    Returns
    -------
    Symbolic variable, free-energy of sample `h1_sample`, at inv. temp beta.
    """
    keep_idx = numpy.arange(not marginalize_odd, model.depth, 2)
    marg_idx = numpy.arange(marginalize_odd, model.depth, 2)

    # contribution of biases
    fe = 0.
    for i in keep_idx:
        fe -= T.dot(samples[i], model.bias[i]) * beta
    # contribution of biases
    for i in marg_idx:
        from_im1 = T.dot(samples[i-1], model.W[i]) if i >= 1 else 0.
        from_ip1 = T.dot(samples[i+1], model.W[i+1].T) if i < model.depth-1 else 0
        net_input = (from_im1 + from_ip1 + model.bias[i]) * beta
        fe -= T.sum(T.nnet.softplus(net_input), axis=1)

    fe -= T.dot(samples[not marginalize_odd], pa_bias) * (1. - beta)

    return fe


def compute_log_ais_weights(model, free_energy_fn, sample_fn, betas):
    """
    Compute log of the AIS weights.
    TODO: remove dependency on global variable model.

    Inputs
    ------
    free_energy_fn: theano.function
        Function which, given temperature beta_k, computes the free energy
        of the samples stored in model.samples. This function should return
        a symbolic vector.
    sample_fn: theano.function
        Function which, given temperature beta_k, generates samples h1 ~
        p_k(h1). These samples are stored in model.nsamples[1].

    Returns
    -------
    log_ais_w: T.vector
        Vector containing log ais-weights.
    """
    # Initialize log-ais weights.
    log_ais_w = numpy.zeros(model.batch_size, dtype=floatX)
    # Iterate from inv. temperature beta_k=0 to beta_k=1...
    for i in range(len(betas) - 1):
        bp, bp1 = betas[i], betas[i+1]
        log_ais_w += free_energy_fn(bp) - free_energy_fn(bp1)
        sample_fn(bp1)
        if i % 1e3 == 0:
            logging.info('Temperature %f ' % bp1)
    return log_ais_w


def estimate_from_weights(log_ais_w):
    """ Safely computes the log-average of the ais-weights.

    Inputs
    ------
    log_ais_w: T.vector
        Symbolic vector containing log_ais_w^{(m)}.

    Returns
    -------
    dlogz: scalar
        log(Z_B) - log(Z_A).
    var_dlogz: scalar
        Variance of our estimator.
    """
    # Utility function for safely computing log-mean of the ais weights.
    ais_w = T.vector()
    max_ais_w = T.max(ais_w)
    dlogz = T.log(T.mean(T.exp(ais_w - max_ais_w))) + max_ais_w
    log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

    # estimate the log-mean of the AIS weights
    dlogz = log_mean(log_ais_w)

    # estimate log-variance of the AIS weights
    # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
    m = numpy.max(log_ais_w)
    var_dlogz = (log_ais_w.shape[0] *
                 numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                 numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

    return dlogz, var_dlogz


def compute_log_za(model, pa_bias, marginalize_odd=True):
    """
    Compute the exact partition function of model p_A(h1).
    """
    log_za = 0.
    
    refa = numpy.sum(numpy.log(1 + numpy.exp(pa_bias)))
    refa += model.n_u[0] * numpy.log(2)
    refa += model.n_u[2] * numpy.log(2)

    for i, nu in enumerate(model.n_u):
        if i == (not marginalize_odd):
            log_za += numpy.sum(numpy.log(1 + numpy.exp(pa_bias)))
        else:
            log_za += numpy.log(2) * nu
    return log_za


def compute_likelihood_given_logz(model, energy_fn, inference_fn, log_z, test_x):
    """
    Compute test set likelihood as below, where q is the variational
    approximation to the posterior p(h1,h2|v).

        ln p(v) \approx \sum_h q(h) E(v,h1,h2) + H(q) - ln Z

    See section 3.2 of DBM paper for details.

    Inputs:
    -------
    model: dbm.DBM
    energy_fn: theano.function
        Function which computes the (temperature 1) energy of the samples stored
        in model.samples. This function should return a symbolic vector.
    inference_fn: theano.function
        Inference function for DBM. Function takes a T.matrix as input (data)
        and returns a list of length `length(model.n_u)`, where the i-th element
        is an ndarray containing approximate samples of layer i.
    log_z: scalar
        Estimate partition function of `model`.
    test_x: numpy.ndarray
        Test set data, in dense design matrix format.

    Returns:
    --------
    Scalar, representing negative log-likelihood of test data under the model.
    """
    i = 0.
    likelihood = 0
    for i in xrange(0, len(test_x), model.batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + model.batch_size, :], dtype=floatX)

        # perform inference
        model.setup_pos_func(x)
        psamples = inference_fn()

        # entropy of h(q) adds contribution to variational lower-bound
        hq = 0
        for psample in psamples[1:]:
            temp = - psample * numpy.log(1e-5 + psample) \
                   - (1.-psample) * numpy.log(1. - psample + 1e-5)
            hq += numpy.sum(temp, axis=1)

        # copy into negative phase buffers to measure energy
        for ii, psample in enumerate(psamples):
            model.nsamples[ii].set_value(psample)

        # compute sum of likelihood for current buffer
        x_likelihood = numpy.sum(-energy_fn(1.0) + hq - log_z)

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        likelihood = (i * likelihood + x_likelihood) / (i + len(x))

    return likelihood


def estimate_likelihood(model, trainset, testset, large_ais=False,
                        log_z=None, seed=980293841):
    """
    Compute estimate of log-partition function and likelihood of data.X.

    Inputs:
    -------
    model: dbm.DBM
    data: pylearn2 dataset
    large_ais: if True, will use 3e5 chains, instead of 3e4
    log_z: log-partition function (if precomputed)

    Returns:
    --------
    nll: scalar
        Negative log-likelihood of data.X under `model`.
    logz: scalar
        Estimate of log-partition function of `model`.
    """
    rng = numpy.random.RandomState(seed)
    theano_rng = RandomStreams(rng.randint(2**30))

    ###
    # Backup DBM states (samples) and uncenter for AIS.
    ###
    dbm_backup = DBMBackup(model)
    dbm_backup.backup()
    dbm_backup.uncenter()

    ##########################
    ## BUILD THEANO FUNCTIONS
    ##########################
    beta = T.scalar()
    
    # for an even number of layers, we marginalize the odd layers (and vice-versa)
    marginalize_odd = (model.depth % 2) == 0
    
    # Build function to retrieve energy.
    E = model.energy(model.nsamples, beta)
    energy_fn = theano.function([beta], E)

    # Build inference function.
    assert (model.pos_mf_steps or model.pos_sample_steps)
    pos_steps = model.pos_mf_steps if model.pos_mf_steps else model.pos_sample_steps
    new_psamples = model.e_step(n_steps=pos_steps)
    inference_fn = theano.function([], new_psamples)

    # Configure baserate bias for (h0 if `marginalize_odd` else h1)
    temp = numpy.asarray(trainset.X, dtype=floatX)
    mean_train = numpy.mean(temp, axis=0)
    model.setup_pos_func(numpy.tile(mean_train[None,:], (model.batch_size,1)))
    psamples = inference_fn()
    mean_pos = numpy.minimum(psamples[not marginalize_odd], 1-1e-5)
    mean_pos = numpy.maximum(mean_pos, 1e-5)
    pa_bias = -numpy.log(1./mean_pos[0] - 1.)

    # Build Theano function to sample from interpolating distributions.
    updates = OrderedDict()
    new_nsamples = neg_sampling(model, model.nsamples, beta=beta,
                                pa_bias=pa_bias,
                                marginalize_odd = marginalize_odd,
                                theano_rng = theano_rng)
    for (nsample, new_nsample) in zip(model.nsamples, new_nsamples):
        updates[nsample] = new_nsample
    sample_fn = theano.function([beta], [], updates=updates, name='sample_func')

    ### Build function to compute free-energy of p_k(h1).
    fe_bp_h1 = free_energy_at_beta(model, model.nsamples, beta,
                                   pa_bias, marginalize_odd = marginalize_odd)
    free_energy_fn = theano.function([beta], fe_bp_h1)


    ###########
    ## RUN AIS
    ###########

    # Generate exact sample for the base model.
    for i, nsample_i in enumerate(model.nsamples):
        bias = pa_bias if i==1 else model.bias[i].get_value()
        hi_mean_vec = 1. / (1. + numpy.exp(-bias))
        hi_mean = numpy.tile(hi_mean_vec, (model.batch_size, 1))
        r = rng.random_sample(hi_mean.shape)
        hi_sample = numpy.array(hi_mean > r, dtype=floatX)
        nsample_i.set_value(hi_sample)

    # default configuration for interpolating distributions
    if large_ais:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e5),
                         numpy.linspace(0.5, 0.9, 1e5),
                         numpy.linspace(0.9, 1.0, 1e5))))
    else:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e4),
                         numpy.linspace(0.5, 0.9, 1e4),
                         numpy.linspace(0.9, 1.0, 1e4))))

    if log_z is None:
        log_ais_w = compute_log_ais_weights(model, free_energy_fn, sample_fn, betas)
        dlogz, var_dlogz = estimate_from_weights(log_ais_w)
        log_za = compute_log_za(model, pa_bias, marginalize_odd)
        log_z = log_za + dlogz
        logging.info('log_z = %f' % log_z)
        logging.info('log_za = %f' % log_za)
        logging.info('dlogz = %f' % dlogz)
        logging.info('var_dlogz = %f' % var_dlogz)

    train_ll = compute_likelihood_given_logz(model, energy_fn, inference_fn, log_z, trainset.X)
    logging.info('Training likelihood = %f' % train_ll)
    test_ll = compute_likelihood_given_logz(model, energy_fn, inference_fn, log_z, testset.X)
    logging.info('Test likelihood = %f' % test_ll)

    ###
    # RESTORE DBM TO ORIGINAL STATE 
    ###
    dbm_backup.restore(recenter=True)

    return (train_ll, test_ll, log_z)


class DBMBackup():
    """
    Backup things which are modified in AIS code.
    """
    def __init__(self, dbm):
        self.dbm = dbm

    def backup(self):
        # backup samples
        self.psamples = [psample.get_value() for psample in self.dbm.psamples]
        self.nsamples = [nsample.get_value() for nsample in self.dbm.nsamples]
        # backup biases
        self.bias = []
        for bias in self.dbm.bias:
            self.bias += [bias.get_value()]

    def restore(self, recenter=True):
        # restore samples
        for (psample, s_psample) in zip(self.dbm.psamples, self.psamples):
            psample.set_value(s_psample)
        for (nsample, s_nsample) in zip(self.dbm.nsamples, self.nsamples):
            nsample.set_value(s_nsample)

        if recenter:
            self.dbm.flags['enable_centering'] = True
            for (bias, s_bias) in zip(self.dbm.bias, self.bias):
                bias.set_value(s_bias)

    def uncenter(self):
        # assume centering for now
        bias = [bias.get_value() for bias in self.dbm.bias]
        offset = [offset.get_value() for offset in self.dbm.offset]
        W = [None] + [W.get_value() for W in self.dbm.W[1:]]
        for i in xrange(self.dbm.depth):
            sub = 0
            # account for offset of lower-layer (if not first layer)
            #bias[i] -=
            sub += numpy.dot(offset[i-1], W[i]) if i > 0 else 0
            # account for offset of upper-layer (if not last layer)
            #bias[i] -=
            sub += numpy.dot(offset[i+1], W[i+1].T) if i < self.dbm.depth - 1 else 0.
            bias[i] -= sub

        self.dbm.flags['enable_centering'] = False
        for i in xrange(self.dbm.depth):
            self.dbm.bias[i].set_value(bias[i])

if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('-m', '--model', action='store', type='string', dest='path')
    parser.add_option('--large', action='store_true', dest='large', default=False)
    parser.add_option('--seed', action='store', type='int', dest='seed', default=980293841)
    (opts, args) = parser.parse_args()

    # Load model and retrieve parameters.
    model = serial.load(opts.path)
    model.do_theano()
    # Load dataset.
    trainset = mnist.MNIST('train', binarize=True)
    testset = mnist.MNIST('test', binarize=True)

    estimate_likelihood(model, trainset, testset, large_ais=opts.large, seed=opts.seed)
