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

import numpy
import pickle
from optparse import OptionParser

import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.gui.patch_viewer import make_viewer

from DBM import sharedX, floatX, npy_floatX

def neg_sampling(model):
    new_nsamples = [model.nsamples[i] for i in xrange(model.depth)]
    for i in xrange(1, model.depth, 2):
        new_nsamples[i] = model.sample_hi_given(new_nsamples, i)
    for i in xrange(0, model.depth, 2):
        if i == 0:
            new_e_nsample0 = model.hi_given(new_nsamples, i)
        else:
            new_nsamples[i] = model.sample_hi_given(new_nsamples, i)
    return new_nsamples, new_e_nsample0


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans', default=1)
parser.add_option('--color', action='store_true',  dest='color')
parser.add_option('--batch_size', action='store',  type='int', dest='batch_size', default=False)
parser.add_option('-n', action='store', dest='n', type='int', default=20)
parser.add_option('--skip', action='store', type='int', dest='skip', default=10)
parser.add_option('--random', action='store_true', dest='random')
parser.add_option('--burnin', action='store', type='int', dest='burnin', default=100)
(opts, args) = parser.parse_args()

# load and recompile model
model = serial.load(opts.path)
model.set_batch_size(opts.batch_size, redo_monitor=False)

###
# Function which computes probability of configuration.
##
energy = model.energy(model.nsamples)
compute_energy = theano.function([], energy)

###
# Rebuild sampling function to have mean-field values for layer 0
##
e_nsamples0 = sharedX(model.nsamples[0].get_value(), name='e_nsamples0')
new_nsamples, new_e_nsample0 = neg_sampling(model)
neg_updates = {e_nsamples0: new_e_nsample0}
for (nsample, new_nsample) in zip(model.nsamples, new_nsamples):
    neg_updates[nsample] = new_nsample
sample_neg_func = theano.function([], [], updates=neg_updates)

if opts.random:
    for nsample in model.nsamples:
        temp = numpy.random.randint(0,2, size=nsample.get_value().shape)
        nsample.set_value(temp.astype(floatX))

# Burnin of Markov chain.
for i in xrange(opts.burnin):
    model.sample_neg_func()

# Start actual sampling.
samples = numpy.zeros((opts.batch_size * opts.n, model.n_u[0]))
indices = numpy.arange(0, len(samples), opts.n)
energies = numpy.zeros(opts.batch_size * opts.n)

for t in xrange(opts.n):
    samples[indices,:] = e_nsamples0.get_value()
    # skip in between plotted samples
    for i in xrange(opts.skip):
        sample_neg_func()
    energies[indices] = compute_energy()
    indices += 1

# transform energies between 0 and 1
energies -= energies.min()
energies /= energies.max()

import pdb; pdb.set_trace()
img = make_viewer(samples,
                  (opts.batch_size, opts.n),
                  (opts.width, opts.height),
                  activation = energies,
                  is_color=opts.color)
img.show()
