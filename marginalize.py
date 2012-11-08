import numpy as np
from operator import itemgetter
from functools import reduce
from itertools import product, ifilter, permutations
from copy import deepcopy

class potential:
    def __init__(self, variable_dict, array):
        self.arities = [v['arity'] for v in sorted(variable_dict.values(),
                                              key=itemgetter('axis'))]
        if tuple(self.arities) != np.shape(array):
            raise ValueError('Dimension mismatch')

        self.variables = sorted(variable_dict.keys(),
                                key=lambda v: variable_dict[v]['axis'])
        self.array = array
        self.variable_dict = variable_dict

    # Given a set of variable values that specifies a single value of the
    # potential function, return that value.
    # Format of the value set is {'X' : val, 'Y' : val...}
    def access(self, var_values):
        value_tuple = tuple(var_values[X] for X in self.variables)
        if len(value_tuple) < len(self.variables):
            raise IndexError('Potential access must be specified with n values')
        return self.array[value_tuple]

    # Return a new potential which is the result of conditioning this
    # potential on the given single observation.
    # Observation format is {'variable' : 'X', 'value' : val}
    def condition(self, observation):
        print self.variable_dict
        axis = self.variable_dict[observation['variable']]['axis']
        value = observation['value']
        slices = []
        for i in range(len(self.variables)):
            if i == axis:
                slices.append(value)
            else:
                slices.append(slice(0, self.arities[i], 1))
        
        new_array = self.array[slices]
        new_variable_dict = deepcopy(self.variable_dict)
        for var in new_variable_dict.values():
            if var['axis'] > axis:
                var['axis'] -= 1

        del(new_variable_dict[observation['variable']])

        return potential(new_variable_dict, new_array)

    # Return a new potential which is the result of marginalizing this
    # potential so as to eliminate the given variable.
    def marginalize(self, X):
        if X not in self.variables:
            raise IndexError('Variable error: X not found in C')
        else:
            C = deepcopy(self.variable_dict)
            i = self.variable_dict[X]['axis']
            for var in C.values():
                if var['axis'] > i:
                    var['axis'] -= 1
            del(C[X])
            marginal_array = np.sum(self.array, axis=i)
            return potential(C, marginal_array)

# Multiply together a list of potentials
def potential_product(potentials_list):
    if not potentials_list:
        return None
    return reduce(potential_pair_product, potentials_list)

# Multiply two potentials into one
def potential_pair_product(phi_X, phi_Y):
    # Produce a list of all the new variables, and indexing arrays
    var_X, var_Y = phi_X.variable_dict, phi_Y.variable_dict
    product_vars = deepcopy(var_X)
    for v in var_Y:
        if v in var_X and var_X[v]['arity'] != var_Y[v]['arity']:
            raise ValueError('Arity mismatch: ' + v)
        else:
            product_vars.update([(v, var_Y[v])])

    # Come up with a new axis ordering for the variables
    a = 0
    product_var_list = []
    for v in product_vars:
        product_vars[v]['axis'] = a
        product_var_list.append(v)
        a += 1

    result_array = np.zeros(shape=tuple(product_vars[k]['arity'] for k in product_var_list))

    # Generate an iterator over all possible values of every variable
    cardinalities = [[(k, i) for i in range(product_vars[k]['arity'])] \
                        for k in product_var_list]
    value_iter = product(*cardinalities)

    for combination in value_iter:
        X_var_values = dict(filter(lambda v: v[0] in var_X, combination))
        Y_var_values = dict(filter(lambda v: v[0] in var_Y, combination))
        X_value = phi_X.access(X_var_values)
        Y_value = phi_Y.access(Y_var_values)
        result_array[tuple(v[1] for v in combination)] = X_value * Y_value

    return potential(product_vars, result_array)

# The variable elimination algorithm
# Takes an elimination ordering, a set of observations, and a set
# of potentials.
# Observations should be in the form
#   {'variable' : 'A', 'value' : 0}
def variable_elimination(potentials, observations, ordering):
    print 'ordering:', ordering
    print
    potentials = list(potentials)
    for E in observations:
        for i in range(len(potentials)):
            p = potentials[i]
            if E['variable'] in p.variables:
                potentials[i] = p.condition(E)

    for X in ordering:
        dependent_potentials = list(filter(lambda p: X in p.variables,
                                           potentials))
        potentials = list(filter(lambda p: X not in p.variables,
                                 potentials))
        if dependent_potentials:
            combined_potential = potential_product(dependent_potentials)
            combined_potential = combined_potential.marginalize(X)
            potentials.append(combined_potential)

    return potential_product(potentials)
