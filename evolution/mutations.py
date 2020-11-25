'''
authors: Wojciech Maciejewski, Christian Konopczyński
'''

import math
import numpy as np

def mutate_one_plus_one(child, cost_function):    
    m = 10.0
    c1 = 0.82
    c2 = 1.2

    sigma = child[-3]
    mutated_child = child.copy()
    for index in range(len(mutated_child) - 3):
        mutated_child[index] += sigma * np.random.normal()
    
    if cost_function(mutated_child) > cost_function(child):
        child = mutated_child
        # increment the number needed for calculating proportion
        child[-2] += 1.0
    
    # increment the iteration number
    child[-1] += 1.0

    if child[-1] == m:
        child[-1] = 0.0
        
        proportion = child[-2] / m
        child[-2] = 0.0
        
        # update sigma
        if proportion < 0.2:
            child[-3] = sigma * c1
        elif proportion > 0.2:
            child[-3] = sigma * c2
        
    return child

def mutate_mu_plus_lambda(child):
    mutated_child = child.copy()
    dimension = int(len(mutated_child)/2)

    tau = 1.0/math.sqrt(2*dimension)
    tau_i = 1.0/math.sqrt(2*math.sqrt(dimension))
    
    norm = np.random.normal()
    for index in range(dimension):
        norm_i = np.random.normal()
        mutated_child[index + dimension] = child[index + dimension] * math.exp(tau * norm + tau_i * norm_i)
        mutated_child[index] = child[index] + mutated_child[index + dimension] * norm_i
    return mutated_child