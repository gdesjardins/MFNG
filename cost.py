import numpy
import theano
import theano.tensor as T
from theano.printing import Print
from collections import OrderedDict
from MFNG import sharedX
floatX = theano.config.floatX

class Cost():

    def __init__(self, cost, params, constants=None):
        self.cost = cost
        self.grads = OrderedDict()
        self.computed_cost = False

        self.params = OrderedDict()
        for p in params:
            self.params[p] = True

        self.constants = OrderedDict()
        constants = [] if constants is None else constants
        for c in constants:
            self.constants[c] = True

    def compute_gradients(self, momentum_lambda=None):
        updates = OrderedDict()
        momentum = OrderedDict()

        grads =  T.grad(self.cost, self.params.keys(), 
                        consider_constant=self.constants.keys(),
                        disconnected_inputs='ignore')
        for param, gparam in zip(self.params.keys(), grads):

            if momentum_lambda:
                momentum[param] = sharedX(numpy.zeros_like(param.get_value()), name=param.name + '_mom')
                new_grad = momentum_lambda * momentum[param] + (1.-momentum_lambda) * gparam
                updates[momentum[param]] = new_grad
            else:
                new_grad = gparam

            self.grads[param] = new_grad

        self.computed_cost = True
        return updates


def compute_gradients(*costs):
    """
    :param args: variable number of grads dictionary
    """

    rval = OrderedDict()
    for cost in costs:
        if not cost.computed_cost:
            cost.compute_gradients()

        for (p,g) in cost.grads.iteritems():
            rval[p] = rval.get(p, 0.) + g

    return rval

def get_cost(*costs):
    total_cost = 0.
    for cost in costs:
        total_cost = cost.cost
    return total_cost


class StalinGrad():
    """
    Scales the gradients by their inverse variance, computed as an exponential moving average
    over time.
    """

    def __init__(self, mov_avg=0.99, eps=1e-3):
        self.mov_avg = mov_avg
        self.eps = eps
        # for each param, stores the moving average of E[x]^2
        self.avg2_x = OrderedDict()
        # for each param, stores the moving average of E[x^2]
        self.avg_x2 = OrderedDict()

    def scale(self, grads):
        """
        :param grads: dictionary of (param,gradient) pairs
        :rval scaled_grad: gradient scaled by inverse variance
        :rval updates: updates dictionary to locally defined shared variables.
        """
        updates = OrderedDict()
        for (param,grad) in grads.iteritems():
            assert isinstance(param, T.sharedvar.TensorSharedVariable)
            pval = param.get_value()
            avg2_x = sharedX(1*numpy.ones_like(pval), name='avg2_x_%s' % param.name)
            avg_x2 = sharedX(10*numpy.ones_like(pval), name='avg_x2_%s' % param.name)
            new_avg2_x = self.mov_avg * avg2_x + (1.-self.mov_avg) * T.mean(grad, axis=0)**2
            new_avg_x2 = self.mov_avg * avg_x2 + (1.-self.mov_avg) * T.mean(grad**2, axis=0)

            grad_var = new_avg_x2 - new_avg2_x

            # scale by inverse of standard deviation, up to max_factor
            #grads[param] = Print('scaler')(T.sqrt(1./(grad_var + self.eps))) * grad
            grads[param] = T.sqrt(1./(grad_var + self.eps)) * grad
           
            # store new shared variables
            self.avg2_x[param] = avg2_x
            self.avg_x2[param] = avg_x2

            # register updates
            updates[avg2_x] = new_avg2_x
            updates[avg_x2] = new_avg_x2

        return grads, updates

def get_updates(grads, lr, fast_lr=None, multipliers=None, momentum_lambda=None):
    """
    Returns an updates dictionary corresponding to a single step of SGD. The learning rate
    for each parameter is computed as lr * multipliers[param]
    :param lr: base learning rate (common to all parameters)
    :param multipliers: dictionary of learning rate multipliers, each being a shared var
                        e.g. {'hbias': sharedX(0.1), 'Wf': sharedX(0.01)}
    """

    updates = OrderedDict()
    momentum = OrderedDict()
    multipliers = OrderedDict() if multipliers is None else multipliers

    for (param, gparam) in grads.iteritems():

        # each parameter can have its own multiplier on the learning rate
        multiplier = multipliers.get(param.name, 1.0)

        if param.name.find('fast')==0 and fast_lr:
            print 'Using fast-learning rate of %f for %s' % (fast_lr, param.name)
            lr_param = fast_lr
        else:
            lr_param = lr * multiplier
   
        # create storage for momentum term
        momentum[param] = sharedX(numpy.zeros_like(param.get_value()), name=param.name + '_old')

        if momentum_lambda and param.name!='fWv':
            # perform SGD, with momentum (optional)
            new_grad = (1.-momentum_lambda) * gparam + momentum_lambda * momentum[param]
            updates[param] = param - lr_param * new_grad
            updates[momentum[param]] = new_grad
        else:
            updates[param] = param - lr_param * gparam

    return updates
