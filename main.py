'''
authors: Wojciech Maciejewski, Christian Konopczyński
'''

import tests
from evolution import models

if __name__ == '__main__':
    
    data = []
    
    for i in range(20):
        data.append(tests.generate_data3())
        
    model = models.Evolutional(cost_function=tests.cost_function3,
                               select_parents='proportional',
                               reproduce='mean',
                               mutate='1+1',
                               select_population='n_best')
        
    history = model.train(starting_population=data,
                          epochs=1000,
                          population_size=100,
                          offspring_size=200,
                          parents_per_child=2)
    
    optimum = model.current_best
    
    memetic_model = models.Memetic(cost_function=tests.cost_function3,
                                   cost_functions_gradient=tests.gradient_for_cost_function3,
                                   select_parents='proportional',
                                   reproduce='mean',
                                   mutate='1+1',
                                   select_population='n_best')
    
    memetic_history = memetic_model.train(starting_population=data,
                                          epochs=1000,
                                          population_size=100,
                                          offspring_size=200,
                                          parents_per_child=2,
                                          memetic_learning_rate=0.1)
    
    memetic_optimum = memetic_model.current_best

