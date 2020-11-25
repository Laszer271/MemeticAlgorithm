'''
authors: Wojciech Maciejewski
'''

import numpy as np

def proportional_select_parents(population, parents_per_child, cost_function):
    
    raw = [cost_function(x) for x in population]
    
    min_raw = min(raw)
    if min_raw < 0:
        raw = [i - min_raw + 1.0 for i in raw]
    
    raw_sum = sum(raw)
    probabilities = [float(i)/raw_sum for i in raw]

    choices = np.random.choice(len(population),
                            parents_per_child,
                            p=probabilities)
    
    return [population[i] for i in choices]