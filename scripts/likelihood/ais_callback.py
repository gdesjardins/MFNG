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

from pylearn2.training_callbacks.training_callback import TrainingCallback

from DBM.scripts.likelihood import ais

class pylearn2_ais_callback(TrainingCallback):

    def __init__(self, trainset, testset,
                 ais_interval=10):

        self.trainset = trainset
        self.testset = testset
        self.ais_interval = ais_interval

        self.pkl_results = {
                'batches_seen': [],
                'cpu_time': [],
                'train_ll': [],
                'test_ll': [],
                'logz': [],
                }

        self.jobman_results = {
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_test_ll': -numpy.Inf,
                'best_logz': 0,
                }
        fp = open('ais_callback.log','w')
        fp.write('Epoch\tBatches\tCPU\tTrain\tTest\tlogz\n')
        fp.close()

    def __call__(self, model, train, algorithm):
        if model.batches_seen == 0:
            return
        if (model.batches_seen % self.ais_interval) != 0:
            return

        (train_ll, test_ll, logz) = ais.estimate_likelihood(model,
                    self.trainset, self.testset, large_ais=False)

        self.log(model, train_ll, test_ll, logz)
        if model.jobman_channel:
            model.jobman_channel.save()

    def log(self, model, train_ll, test_ll, logz):

        # log to database
        self.jobman_results['batches_seen'] = model.batches_seen
        self.jobman_results['cpu_time'] = model.cpu_time
        self.jobman_results['train_ll'] = train_ll
        self.jobman_results['test_ll'] = test_ll
        self.jobman_results['logz'] = logz
        if train_ll > self.jobman_results['best_train_ll']:
            self.jobman_results['best_batches_seen'] = self.jobman_results['batches_seen']
            self.jobman_results['best_cpu_time'] = self.jobman_results['cpu_time']
            self.jobman_results['best_train_ll'] = self.jobman_results['train_ll']
            self.jobman_results['best_test_ll'] = self.jobman_results['test_ll']
            self.jobman_results['best_logz'] = self.jobman_results['logz']
        model.jobman_state.update(self.jobman_results)

        # save to text file
        fp = open('ais_callback.log','a')
        fp.write('%i\t%f\t%f\t%f\t%f\n' % (
            self.jobman_results['batches_seen'],
            self.jobman_results['cpu_time'],
            self.jobman_results['train_ll'],
            self.jobman_results['test_ll'],
            self.jobman_results['logz']))
        fp.close()

        # save to pickle file
        self.pkl_results['batches_seen'] += [model.batches_seen]
        self.pkl_results['cpu_time'] += [model.cpu_time]
        self.pkl_results['train_ll'] += [train_ll]
        self.pkl_results['test_ll'] += [test_ll]
        self.pkl_results['logz'] += [logz]
        fp = open('ais_callback.pkl','w')
        pickle.dump(self.pkl_results, fp)
        fp.close()


