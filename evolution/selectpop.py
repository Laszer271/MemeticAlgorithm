'''
authors: Wojciech Maciejewski, Christian Konopczyński
'''
import numpy as np

def n_best_select(data, count, cost_function):
    sorted_data = data.copy()
    sorted_data.sort(key=cost_function, reverse=True)
    return sorted_data[: count]
'''
def roulette_wheel_select(data, count, cost_function):
    fitness_sum = sum([cost_function(datum) for datum in data])
    previous_probability = 0.0
    probabilities = np.array([cost_function(datum)/fitness_sum for datum in data])
    sorted_probabilities = probabilities.copy()
    sorted_probabilities.sort()
    
    for p in sorted_probabilities:
        temp = p
        p += previous_probability
        previous_probability = temp

    result = []
    
    for i in count:
        result.append(p.searchsorted(sorted_probabilities, np.random.uniform(), 'right'))

    return result
'''
