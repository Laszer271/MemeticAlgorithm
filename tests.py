import random
import numpy as np

def generate_data1():
    params = []
    params.append(random.uniform(-100.0, 100.0))
    params.append(random.uniform(-5.0, 5.0))

    return params

def example_cost_function1(parameters):
    return -1.0 * parameters[0]**2 + parameters[0] - 1.0

def gradient_for_example_function1(parameters):
    return np.array([-2.0 * parameters[0] + 1.0,
                     0])

def generate_data2():
    params = []
    for i in range(12):
        params.append(random.uniform(-5.0, 5.0))
        
    #sigma for 1+1 algorithm
    params.append(1.0)
    #proportion (fi) for 1+1 algorithm
    params.append(0.0)
    #iteration counter
    params.append(0)
    
    correct_function(params)
    return params

def cost_function2(parameters):
    result = parameters[1]
    powered_x = parameters[0]
    for i in range(2, 12):
        result += powered_x * parameters[i]
        powered_x *= parameters[0]
    return result

def gradient_for_cost_function2(parameters):
    gradient = []
    powered_x = parameters[0]
    result = parameters[2]
    for i in range(3, 12):
        result += powered_x * parameters[i]
        powered_x *= parameters[0]

    gradient.append(result)
    gradient.append(0.0)
    powered_x = parameters[0]
    for i in range(2, 12):
        gradient.append(powered_x)
        powered_x *= parameters[0]
        
    gradient.extend([0.0, 0.0, 0.0])
    return np.array(gradient)

def correct_function(parameters):
    if parameters[11] > 0.0:
        parameters[11] *= -1.0
    for i in range(12):
        if parameters[i] > 5.0:
            parameters[i] = 5.0
        if parameters[i] < -5.0:
            parameters[i] = -5.0
            
    return parameters

def generate_data3():
    params = []
    for i in range(20):
        params.append(random.uniform(-5.0, 5.0))
        
    #sigma for 1+1 algorithm
    params.append(1.0)
    #proportion (fi) for 1+1 algorithm
    params.append(0.0)
    #iteration counter
    params.append(0)

    return params

def cost_function3(parameters):
    result = 1.0
    for parameter in parameters[: len(parameters) - 3]:
        result -= parameter ** 2
        
    return result

def gradient_for_cost_function3(parameters):
    gradient = []
    for parameter in parameters[: len(parameters) - 3]:
        gradient.append(-2.0 * parameter)
    
    gradient.extend([0.0, 0.0, 0.0])
    
    return np.array(gradient)