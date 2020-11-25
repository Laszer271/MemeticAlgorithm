'''
authors: Wojciech Maciejewski
'''

import numpy as np
import math

def mean_reproduce(parents):
    child = np.mean(parents, axis=0)
    
    child[-1] = math.ceil(child[-1])
    
    return child
