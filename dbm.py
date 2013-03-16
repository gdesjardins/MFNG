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
import os
import numpy
import pickle
import time
from collections import OrderedDict
from scipy import stats

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from theano.sandbox.scan import scan

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

from MFNG import tools
from MFNG import cost as utils_cost
from MFNG import lincg
from MFNG import minres
from MFNG import fisher
from MFNG import utils
from MFNG import sharedX, floatX, npy_floatX
from theano_optimize import minresQLP

class MFNG(Model, Block):
    """Bilinear Restricted Boltzmann Machine (RBM)  """

    def __init__(self, input = None, n_u=[100,100], enable={}, load_from=None,
            iscales=None, clip_min={}, clip_max={},
            pos_mf_steps=1, pos_sample_steps=0, neg_sample_steps=1, 
            lr_spec={}, lr_mults = {},
            l1 = {}, l2 = {}, l1_inf={}, flags={}, momentum_lambda=0,
            cg_params = {},
            batch_size = 13,
            computational_bs = 0,
            compile=True,
            seed=1241234,
            sp_targ_h = None, sp_weight_h=None, sp_pos_k = 5,
            my_save_path=None, save_at=None, save_every=None,
            max_updates=1e6):
        """
        :param n_u: list, containing number of units per layer. n_u[0] contains number
         of visible units, while n_u[i] (with i > 0) contains number of hid. units at layer i.
        :param enable: dictionary of flags with on/off behavior
        :param iscales: optional dictionary containing initialization scale for each parameter.
               Key of dictionary should match the name of the associated shared variable.
        :param pos_mf_steps: number of mean-field iterations to perform in positive phase
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param lr: base learning rate
        :param lr_timestamp: list containing update indices at which to change the lr multiplier
        :param lr_mults: dictionary, optionally containing a list of learning rate multipliers
               for parameters of the model. Length of this list should match length of
               lr_timestamp (the lr mult will transition whenever we reach the associated
               timestamp). Keys should match the name of the shared variable, whose learning
               rate is to be adjusted.
        :param l1: dictionary, whose keys are model parameter names, and values are
               hyper-parameters controlling degree of L1-regularization.
        :param l2: same as l1, but for L2 regularization.
        :param l1_inf: same as l1, but the L1 penalty is centered as -\infty instead of 0.
        :param cg_params: dictionary with keys ['rtol','damp','maxiter']
        :param batch_size: size of positive and negative phase minibatch
        :param computational_bs: batch size used internaly by natural
               gradient to reduce memory consumption
        :param seed: seed used to initialize numpy and theano RNGs.
        :param my_save_path: if None, do not save model. Otherwise, contains stem of filename
               to which we will save the model (everything but the extension).
        :param save_at: list containing iteration counts at which to save model
        :param save_every: scalar value. Save model every `save_every` iterations.
        """
        Model.__init__(self)
        Block.__init__(self)
        ### VALIDATE PARAMETERS AND SET DEFAULT VALUES ###
        assert lr_spec is not None
        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)
        [iscales.setdefault('bias%i' % i, 0.) for i in xrange(len(n_u))]
        [iscales.setdefault('W%i' % i, 0.1) for i in xrange(len(n_u))]
        flags.setdefault('enable_centering', False)
        flags.setdefault('enable_natural', False)
        flags.setdefault('enable_warm_start', False)
        flags.setdefault('mlbiases', False)
        flags.setdefault('precondition', None)
        flags.setdefault('minres', False)
        flags.setdefault('minresQLP', False)
        if flags['precondition'] == 'None': flags['precondition'] = None
       
        self.jobman_channel = None
        self.jobman_state = {}
        self.register_names_to_del(['jobman_channel'])

        ### DUMP INITIALIZATION PARAMETERS TO OBJECT ###
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        assert len(n_u) > 1
        self.n_v = n_u[0]
        self.depth = len(n_u)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed)
        self.theano_rng = RandomStreams(self.rng.randint(2**30))

        # allocate bilinear-weight matrices
        self.input = T.matrix()
        self.init_parameters()
        self.init_dparameters()
        self.init_centering()
        self.init_samples()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr_spec['type'] == 'anneal':
            num = lr_spec['init'] * lr_spec['start'] 
            denum = T.maximum(lr_spec['start'], lr_spec['slope'] * self.iter)
            self.lr = T.maximum(lr_spec['floor'], num/denum) 
        elif lr_spec['type'] == 'linear':
            lr_start = npy_floatX(lr_spec['start'])
            lr_end   = npy_floatX(lr_spec['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        else:
            raise ValueError('Incorrect value for lr_spec[type]')

        # counter for CPU-time
        self.cpu_time = 0.

        if load_from:
            self.load_parameters(fname=load_from)

        # configure input-space (?new pylearn2 feature?)
        self.input_space = VectorSpace(n_u[0])
        self.output_space = VectorSpace(n_u[-1])
        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size
        self.error_record = []
 
        if compile: self.do_theano()

    def init_parameters(self):
        # Create shared variables for model parameters.
        self.W = []
        self.bias = []
        for i, nui in enumerate(self.n_u):
            self.bias += [sharedX(self.iscales['bias%i' %i] * numpy.ones(nui), name='bias%i'%i)]
            self.W += [None]
            if i > 0: 
                wv_val = self.rng.randn(self.n_u[i-1], nui) * self.iscales.get('W%i'%i,1.0)
                self.W[i] = sharedX(wv_val, name='W%i' % i)
        # Establish list of learnt model parameters.
        self.params  = [Wi for Wi in self.W[1:]]
        self.params += [bi for bi in self.bias]

    # pylearn2 compatibility
    def get_params(self):
        return self.params

    def load_parameters(self, fname):
        # load model
        fp = open(fname)
        model = pickle.load(fp)
        fp.close()
        # overwrite local parameters
        for (m_wi, wi) in zip(model.W[1:], self.W[1:]):
            wi.set_value(m_wi.get_value())
        for (m_bi, bi) in zip(model.bias, self.bias):
            bi.set_value(m_bi.get_value())
        for (m_offi, offi) in zip(model.offset, self.offset):
            offi.set_value(m_offi.get_value())
        self.batches_seen = model.batches_seen
        self.epochs = model.epochs
        # load negative phase particles
        mi = 0
        for k in xrange(self.depth):
            nsamples_k = self.nsamples[k].get_value()
            m_nsamples_k = model.nsamples[k].get_value()
            for i in xrange(self.batch_size):
                nsamples_k[i,:] = m_nsamples_k[mi, :]
                mi = (mi + 1) % model.batch_size
            self.nsamples[k].set_value(nsamples_k)

    def init_dparameters(self):
        # Create shared variables for model parameters.
        self.dW = []
        self.dbias = []
        for i, nui in enumerate(self.n_u):
            self.dbias += [sharedX(numpy.zeros(nui), name='dbias%i'%i)]
            self.dW += [None]
            if i > 0: 
                wv_val = numpy.zeros((self.n_u[i-1], nui))
                self.dW[i] = sharedX(wv_val, name='dW%i' % i)
        self.dparams  = [dWi for dWi in self.dW[1:]]
        self.dparams += [dbi for dbi in self.dbias]
 
    def init_centering(self):
        self.offset = []
        for i, nui in enumerate(self.n_u):
            self.offset += [sharedX(numpy.zeros(nui), name='offset%i'%i)]

    def init_samples(self):
        self.psamples = []
        self.nsamples = []
        for i, nui in enumerate(self.n_u):
            self.psamples += [sharedX(self.rng.rand(self.batch_size, nui), name='psamples%i'%i)]
            self.nsamples += [sharedX(self.rng.rand(self.batch_size, nui), name='nsamples%i'%i)]

    def setup_pos(self):
        updates = OrderedDict()
        updates[self.psamples[0]] = self.input
        for i in xrange(1, self.depth):
            if self.flags['enable_centering']:
                layer_init = T.ones((self.input.shape[0], self.n_u[i])) * self.offset[i]
            else:
                layer_init = 0.5 * T.ones((self.input.shape[0], self.n_u[i]))
            updates[self.psamples[i]] = layer_init
        return theano.function([self.input], [], updates=updates)

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""
        self.flags.setdefault('enable_warm_start', False)

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        ###
        # FUNCTION WHICH PREPS POS PHASE
        ###
        self.setup_pos_func = self.setup_pos()
 
        ###
        # POSITIVE PHASE ESTEP
        ###
        if self.pos_mf_steps:
            assert self.pos_sample_steps == 0
            new_psamples = self.e_step(n_steps=self.pos_mf_steps)
        else:
            new_psamples = self.pos_sampling(n_steps=self.pos_sample_steps)
        pos_updates = self.e_step_updates(new_psamples)
        self.pos_func = function([], [], updates=pos_updates, name='pos_func', profile=0)

        ###
        # SAMPLING: NEGATIVE PHASE
        ###
        new_nsamples = self.neg_sampling(self.nsamples)
        new_ev = self.hi_given(new_nsamples, 0)
        neg_updates = OrderedDict()
        for (nsample, new_nsample) in zip(self.nsamples, new_nsamples):
            neg_updates[nsample] = new_nsample
        self.sample_neg_func = function([], [], updates=neg_updates,
                                        name='sample_neg_func', profile=0)

        ###
        # SML LEARNING
        ###
        ml_cost = self.ml_cost(self.psamples, self.nsamples)
        mom_updates = ml_cost.compute_gradients()
        reg_cost = self.get_reg_cost()
        #sp_cost = self.get_sparsity_cost()
        cg_output = []
        natgrad_updates = OrderedDict()
        if self.flags['enable_natural']:
            xinit = self.dparams if self.flags['enable_warm_start'] else None
            cg_output, natgrad_updates = self.get_natural_direction(
                    ml_cost, self.nsamples,
                    xinit = xinit,
                    precondition = self.flags.get('precondition',None))
        elif self.flags['enable_natural_diag']:
            cg_output, natgrad_updates = self.get_natural_diag_direction(ml_cost, self.nsamples)
        learning_grads = utils_cost.compute_gradients(ml_cost, reg_cost)

        ##
        # COMPUTE GRADIENTS WRT. TO ALL COSTS
        ##
        learning_updates = utils_cost.get_updates(             
                learning_grads,
                self.lr,
                multipliers = self.lr_mults,
                momentum_lambda = self.momentum_lambda) 
        learning_updates.update(natgrad_updates)
        learning_updates.update(mom_updates)
        learning_updates.update({self.iter: self.iter+1})
      
        # build theano function to train on a single minibatch
        self.batch_train_func = function([], cg_output,
                updates=learning_updates,
                name='train_rbm_func',
                profile=0)

        ##
        # CONSTRAINTS
        ##
        constraint_updates = OrderedDict()

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params]
            for p in self.params:
                if p.name == k:
                    break
            constraint_updates[p] = T.clip(constraint_updates.get(p, p), v, p)

        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()


    def train_batch(self, dataset, batch_size):
        """
        Performs one-step of gradient descent, using the given dataset.
        :param dataset: Pylearn2 dataset to train the model with.
        :param batch_size: int. Batch size to use.
               HACK: this has to match self.batch_size.
        """
        # First-layer biases of RBM-type models should always be initialized to the log-odds
        # ratio. This ensures that the weights don't attempt to learn the mean.
        if self.flags['mlbiases'] and self.batches_seen == 0:
            # set layer 0 biases
            mean_x = numpy.mean(dataset.X, axis=0)
            clip_x = numpy.clip(mean_x, 1e-5, 1-1e-5)
            self.bias[0].set_value(numpy.log(clip_x / (1. - clip_x)))
            for i in xrange(self.depth):
                offset_i = 1./(1 + numpy.exp(-self.bias[i].get_value()))
                self.offset[i].set_value(offset_i)

        x = dataset.get_batch_design(batch_size, include_labels=False)
        self.learn_mini_batch(x)
        self.enforce_constraints()

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        # save to different path each epoch
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):
            fname = self.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, self)
            print 'done'

        return self.batches_seen < self.max_updates

    def learn_mini_batch(self, x):
        """
        Performs the substeps involed in one iteration of PCD/SML. We first adapt the learning
        rate, generate new negative samples from our persistent chain and then perform a step
        of gradient descent.
        :param x: numpy.ndarray. mini-batch of training examples, of shape (batch_size, self.n_u[0])
        """
        # perform variational/sampling positive phase
        t1 = time.time()
        self.setup_pos_func(x)
        self.pos_func()
        for i in xrange(self.neg_sample_steps):
            self.sample_neg_func()
        rval = self.batch_train_func()
        self.cpu_time += time.time() - t1

        ### LOGGING & DEBUGGING ###
        if len(rval) and self.batches_seen%100 == 0:
            fp = open('cg.log', 'a' if self.batches_seen else 'w')
            fp.write('Batches: %i\t niters:%i\t rk_res:%s\t mcos_dist=%s\n' %
                     (self.batches_seen, rval[0], str(rval[1]), str(rval[2])))
            fp.close()

    def center_samples(self, samples):
        if self.flags['enable_centering']:
            return [samples[i] - self.offset[i] for i in xrange(len(samples))]
        else:
            return samples

    def energy(self, samples, beta=1.0, alpha=0.):
        """
        Computes energy for a given configuration of visible and hidden units.
        :param samples: list of T.matrix of shape (batch_size, n_u[i])
        samples[0] represents visible samples.
        """
        csamples = self.center_samples(samples)
        energy = - T.dot(csamples[0], self.bias[0]) * beta
        for i in xrange(1, self.depth):
            energy -= T.sum(T.dot(csamples[i-1], self.W[i] * beta) * csamples[i], axis=1)
            energy -= T.dot(csamples[i], self.bias[i] * beta)

        return energy

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def hi_given(self, samples, i, beta=1.0, apply_sigmoid=True):
        """
        Compute the state of hidden layer i given all other layers.
        :param samples: list of tensor-like objects. For the positive phase, samples[0] is
        points to self.input, while samples[i] contains the current state of the i-th layer. In
        the negative phase, samples[i] contains the persistent chain associated with the i-th
        layer.
        :param i: int. Compute activation of layer i of our DBM.
        :param beta: used when performing AIS.
        :param apply_sigmoid: when False, hi_given will not apply the sigmoid. Useful for AIS
        estimate.
        """
        csamples = self.center_samples(samples)

        hi_mean = 0.
        if i < self.depth-1:
            # top-down input
            wip1 = self.W[i+1]
            hi_mean += T.dot(csamples[i+1], wip1.T) * beta

        if i > 0:
            # bottom-up input
            wi = self.W[i]
            hi_mean += T.dot(csamples[i-1], wi) * beta

        hi_mean += self.bias[i] * beta

        if apply_sigmoid:
            return T.nnet.sigmoid(hi_mean)
        else:
            return hi_mean

    def sample_hi_given(self, samples, i, beta=1.0):
        """
        Given current state of our DBM (`samples`), sample the values taken by the i-th layer.
        See self.hi_given for detailed description of parameters.
        """
        hi_mean = self.hi_given(samples, i, beta)

        hi_sample = self.theano_rng.binomial(
                size = (self.batch_size, self.n_u[i]),
                n=1, p=hi_mean, 
                dtype=floatX)

        return hi_sample

 
    ##################
    # SAMPLING STUFF #
    ##################

    def pos_sampling(self, n_steps=50):
        """
        Performs `n_steps` of mean-field inference (used to compute positive phase statistics).
        :param psamples: list of tensor-like objects, representing the state of each layer of
        the DBM (during the inference process). psamples[0] points to self.input.
        :param n_steps: number of iterations of mean-field to perform.
        """
        new_psamples = [T.unbroadcast(T.shape_padleft(psample)) for psample in self.psamples]

        # now alternate mean-field inference for even/odd layers
        def sample_iteration(*psamples):
            new_psamples = [p for p in psamples]
            for i in xrange(1,self.depth,2):
                new_psamples[i] = self.sample_hi_given(psamples, i)
            for i in xrange(2,self.depth,2):
                new_psamples[i] = self.sample_hi_given(psamples, i)
            return new_psamples

        new_psamples, updates = scan(
                sample_iteration,
                states = new_psamples,
                n_steps=n_steps)

        return [x[0] for x in new_psamples]

    def e_step(self, n_steps=100, eps=1e-5):
        """
        Performs `n_steps` of mean-field inference (used to compute positive phase statistics).
        :param psamples: list of tensor-like objects, representing the state of each layer of
        the DBM (during the inference process). psamples[0] points to self.input.
        :param n_steps: number of iterations of mean-field to perform.
        """
        new_psamples = [T.unbroadcast(T.shape_padleft(psample)) for psample in self.psamples]

        # now alternate mean-field inference for even/odd layers
        def mf_iteration(*psamples):
            new_psamples = [p for p in psamples]
            for i in xrange(1,self.depth,2):
                new_psamples[i] = self.hi_given(psamples, i)
            for i in xrange(2,self.depth,2):
                new_psamples[i] = self.hi_given(psamples, i)

            score = 0.
            for i in xrange(1, self.depth):
                score = T.maximum(T.mean(abs(new_psamples[i] - psamples[i])), score)

            return new_psamples, theano.scan_module.until(score < eps)

        new_psamples, updates = scan(
                mf_iteration,
                states = new_psamples,
                n_steps=n_steps)

        return [x[0] for x in new_psamples]

    def e_step_updates(self, new_psamples):
        updates = OrderedDict()
        for (new_psample, psample) in zip(new_psamples, self.psamples):
            updates[psample] = new_psample
        return updates

    def neg_sampling(self, nsamples, beta=1.0):
        """
        Perform `n_steps` of block-Gibbs sampling (used to compute negative phase statistics).
        This method alternates between sampling of odd given even layers, and vice-versa.
        :param nsamples: list (of length len(self.n_u)) of tensor-like objects, representing
        the state of the persistent chain associated with layer i.
        """
        new_nsamples = [nsamples[i] for i in xrange(self.depth)]
        for i in xrange(1,self.depth,2):
            new_nsamples[i] = self.sample_hi_given(new_nsamples, i, beta)
        for i in xrange(0,self.depth,2):
            new_nsamples[i] = self.sample_hi_given(new_nsamples, i, beta)
        return new_nsamples

    def ml_cost(self, psamples, nsamples):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_cost = T.sum(self.energy(psamples))
        neg_cost = T.sum(self.energy(nsamples))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size

        cte = psamples + nsamples
        return utils_cost.Cost(cost, self.params, cte)

    def monitor_stats(self, b, axis=(0,1), name=None, track_min=True, track_max=True):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name
        channels = {name + '.mean': T.mean(b, axis=axis)}
        if track_min: channels[name + '.min'] = T.min(b, axis=axis)
        if track_max: channels[name + '.max'] = T.max(b, axis=axis)
        return channels

    def get_monitoring_channels(self, x, y=None):
        chans = {}
        chans['lr'] = self.lr
        chans['iter'] = self.iter

        cpsamples = self.center_samples(self.psamples)
        cnsamples = self.center_samples(self.nsamples)

        for i in xrange(self.depth):
            chans.update(self.monitor_stats(self.bias[i], axis=(0,)))
            chans.update(self.monitor_stats(self.psamples[i]))
            chans.update(self.monitor_stats(self.nsamples[i]))
            chans.update(self.monitor_stats(cpsamples[i], name='cpsamples%i'%i))
            chans.update(self.monitor_stats(cnsamples[i], name='cnsamples%i'%i))

        for i in xrange(1, self.depth):
            chans.update(self.monitor_stats(self.W[i]))
            norm_wi = T.sqrt(T.sum(self.W[i]**2, axis=0))
            chans.update(self.monitor_stats(norm_wi, axis=(0,), name='norm_w%i'%i))

        def normalize(x):
            return x / T.sqrt(T.sum(x**2))

        return chans

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    """
    def get_sparsity_cost(self):

        # update mean activation using exponential moving average
        posh = self.e_step(self.psamples, self.sp_pos_k)

        # define loss based on value of sp_type
        eps = 1./self.batch_size
        loss = lambda targ, val: - targ * T.log(eps + val) - (1.-targ) * T.log(1. - val + eps)

        cost = T.zeros((), dtype=floatX)
        params = []
        if self.sp_weight_h:
            for (i, poshi) in enumerate(posh):
                cost += self.sp_weight_h  * T.sum(loss(self.sp_targ_h, poshi.mean(axis=0)))
                if self.W[i]: params += [self.W[i]]
                if self.bias[i]: params += [self.bias[i]]

        return utils_cost.Cost(cost, params)
    """


    def get_reg_cost(self):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict containing amount of L2 regularization for Wg, Wh and Wv
        :param l1: dict containing amount of L1 regularization for Wg, Wh and Wv
        :param l1_inf: dict containing amount of L1 (centered at -inf) reg for Wg, Wh and Wv
        """
        cost = 0.
        params = []

        for p in self.params:

            if self.l1.has_key(p.name):
                cost += self.l1[p.name] * T.sum(abs(p))
                params += [p]

            if self.l1_inf.has_key(p.name):
                cost += self.l1_inf[p.name] * T.sum(p)
                params += [p]

            if self.l2.has_key(p.name):
                cost += self.l2[p.name] * T.sum(p**2)
                params += [p]

        return utils_cost.Cost(cost, params)

    def get_dparam_updates(self, *deltas):
        updates = OrderedDict()
        if self.flags['enable_warm_start']:
            updates[self.dW[1]] = deltas[0]
            updates[self.dW[2]] = deltas[1]
            updates[self.dbias[0]] = deltas[2]
            updates[self.dbias[1]] = deltas[3]
            updates[self.dbias[2]] = deltas[4]
        return updates

    def get_natural_diag_direction(self, ml_cost, nsamples):
        damp = self.cg_params['damp']
        cnsamples = self.center_samples(nsamples)
        rvals = fisher.compute_L_diag(cnsamples)

        # keep track of cosine similarity
        cos_dist  = 0.
        norm2_old = 0.
        norm2_new = 0.
        for i, param in enumerate(self.params):
            new_gradi = ml_cost.grads[param] * 1./(rvals[i] + damp)
            norm2_old += T.sum(ml_cost.grads[param]**2)
            norm2_new += T.sum(new_gradi**2)
            cos_dist += T.dot(ml_cost.grads[param].flatten(), new_gradi.flatten())
            ml_cost.grads[param] = new_gradi
        cos_dist /= (norm2_old * norm2_new)

        return [T.constant(1), T.constant(0), cos_dist], OrderedDict()

    def get_natural_direction(self, ml_cost, nsamples, xinit=None,
                              precondition=None):
        """
        Returns: list
            See lincg documentation for the meaning of each return value.
            rvals[0]: niter
            rvals[1]: rerr
        """
        assert precondition in [None, 'jacobi']
        self.cg_params.setdefault('batch_size', self.batch_size)

        nsamples = nsamples[:self.cg_params['batch_size']]
        neg_energies = self.energy(nsamples)

        if self.computational_bs > 0:
            raise NotImplementedError()
        else:
            def Lx_func(*args):
                Lneg_x = fisher.compute_Lx(
                        neg_energies,
                        self.params,
                        args)
                if self.flags['minresQLP']:
                    return Lneg_x, {}
                else:
                    return Lneg_x

        M = None
        if precondition == 'jacobi':
            cnsamples = self.center_samples(nsamples)
            raw_M = fisher.compute_L_diag(cnsamples)
            M = [(Mi + self.cg_params['damp']) for Mi in raw_M]

        if self.flags['minres']:
            rvals = minres.minres(
                    Lx_func,
                    [ml_cost.grads[param] for param in self.params],
                    rtol = self.cg_params['rtol'],
                    maxiter = self.cg_params['maxiter'],
                    damp = self.cg_params['damp'],
                    xinit = xinit,
                    Ms = M)
            [newgrads, flag, niter, rerr] = rvals[:4]
        elif self.flags['minresQLP']:
            param_shapes = []
            for p in self.params:
                param_shapes += [p.get_value().shape]
            rvals = minresQLP.minresQLP(
                    Lx_func,
                    [ml_cost.grads[param] for param in self.params],
                    param_shapes,
                    rtol = self.cg_params['rtol'],
                    maxit = self.cg_params['maxiter'],
                    damp = self.cg_params['damp'],
                    Ms = M,
                    profile = 0)
            [newgrads, flag, niter, rerr] = rvals[:4]
        else:
            rvals = lincg.linear_cg(
                    Lx_func,
                    [ml_cost.grads[param] for param in self.params],
                    rtol = self.cg_params['rtol'],
                    damp = self.cg_params['damp'],
                    maxiter = self.cg_params['maxiter'],
                    xinit = xinit,
                    M = M)
            [newgrads, niter, rerr] = rvals

        # Now replace grad with natural gradient.
        cos_dist  = 0.
        norm2_old = 0.
        norm2_new = 0.
        for i, param in enumerate(self.params):
            norm2_old += T.sum(ml_cost.grads[param]**2)
            norm2_new += T.sum(newgrads[i]**2)
            cos_dist += T.dot(ml_cost.grads[param].flatten(),
                              newgrads[i].flatten())
            ml_cost.grads[param] = newgrads[i]
        cos_dist /= (norm2_old * norm2_new)
        
        return [niter, rerr, cos_dist], self.get_dparam_updates(*newgrads)

    def switch_to_full_natural(self):
        self.flags['enable_natural'] = True
        self.flags['enable_natural_diag'] = False
        self.set_batch_size(256)

    def set_batch_size(self, batch_size, redo_monitor=True):
        """
        Change the batch size of a model which has already been initialized.
        :param batch_size: int. new batch size.
        """
        # re-allocate shared variables
        for k in xrange(self.depth):
            new_psample = numpy.zeros((batch_size, self.n_u[k])).astype(floatX)
            self.psamples[k].set_value(new_psample)

            # preserve negative phase particles
            new_nsample = numpy.zeros((batch_size, self.n_u[k])).astype(floatX)
            old_nsample = self.nsamples[k].get_value()
            mi = 0
            for i in xrange(batch_size):
                new_nsample[i,:] = old_nsample[mi, :]
                mi = (mi + 1) % self.batch_size
            self.nsamples[k].set_value(new_nsample)

        self.batch_size = batch_size
        self.force_batch_size = batch_size
        self.do_theano()
        for i in xrange(len(self.monitor._batch_size)):
            self.monitor._batch_size[i] = batch_size

        if redo_monitor:
            self.monitor.redo_theano()

    def __call__(self, v):
        return T.horizontal_stack(*self.psamples[1:])
