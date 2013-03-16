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

import sys
import numpy
import pylab as pl
import pickle
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--global',  action='store_false', dest='local',    default=True)
parser.add_option('--preproc', action='store', type='string', dest='preproc')
parser.add_option('-k', action='store', type='int', dest='k', default=-1)
(opts, args) = parser.parse_args()

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1

# load model and retrieve parameters
model = serial.load(opts.path)

##############
# PLOT FILTERS
##############

def get_dims(n):
    num_rows = numpy.floor(numpy.sqrt(n))
    return (numpy.int(num_rows),
            numpy.int(numpy.ceil(n / num_rows)))

nblocks = model.depth - 1
W = [model.W[i].get_value().T for i in xrange(1, model.depth)]
max_filters = max([len(Wi) for Wi in W])
print 'max_filters = ', max_filters

block_viewer = PatchViewer(get_dims(max_filters),
                           (opts.height, opts.width), 
                           is_color = opts.color,
                           pad=(2,2))

main_viewer = PatchViewer(get_dims(nblocks), 
                          (block_viewer.image.shape[0],
                           block_viewer.image.shape[1]),
                          is_color = opts.color, 
                          pad=(5,5))

topo_shape = [opts.height, opts.width, opts.chans]
view_converter = DefaultViewConverter(topo_shape)

for di, w_di in enumerate(W):

    if opts.k == -1:
        # build "new_w" as linear combination of all previous filters
        if di > 0:
            new_w = numpy.dot(w_di, prev_w)
        else: 
            new_w = w_di
    else:
        new_w = numpy.zeros((len(w_di), opts.height * opts.width)) if di else w_di

    for fi in xrange(len(w_di)):

        if opts.k != -1:
            # build "new_w" as a linear combination of "strongest" filters in layer below
            if di > 0:
                temp.fill(0.)
                idx = numpy.argsort(w_di[fi])[-opts.k:]
                for fi_m1 in idx:
                    new_w[fi:fi+1] += w_di[fi, fi_m1] * prev_w[fi_m1:fi_m1+1,:]
                #for fi_m1 in xrange(len(w_di[fi])):
            else:
                temp = w_di[fi:fi+1,:]

        topo_view = view_converter.design_mat_to_topo_view(new_w[fi:fi+1])
        block_viewer.add_patch(topo_view[0])

    main_viewer.add_patch(block_viewer.image[:,:,0] - 0.5)
    block_viewer.clear()
    
    prev_w = new_w

pl.imshow(main_viewer.image, interpolation=None)
pl.axis('off');
pl.savefig('weights.png')
pl.show()
