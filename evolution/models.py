'''
authors: Wojciech Maciejewski, Christian Konopczyński
'''

from . import selectparents
from . import crossovers
from . import mutations
from . import selectpop

import numpy as np
import sys

class Evolutional:
    
    def __init__(self, cost_function, select_parents, reproduce, mutate,
                 select_population, correct=None):
        
        self.cost_function = cost_function
        
        self.select_parents = self.choose_select_parents_fun(select_parents)
        self.reproduce = self.choose_reproduce_fun(reproduce)
        self.mutate = self.choose_mutate_fun(mutate)
        self.select_population = self.choose_select_population_fun(select_population)
        self.correct = correct
        
    @staticmethod
    def choose_select_parents_fun(fun_name):
        if fun_name == 'proportional':
            fun = selectparents.proportional_select_parents
        else: 
            raise RuntimeError(f'there is no such select_parents function as {fun_name} available in this library')
            
        return fun
      
    @staticmethod
    def choose_reproduce_fun(fun_name):
            if fun_name == 'mean':
                fun = crossovers.mean_reproduce
            else: 
                raise RuntimeError(f'there is no such reproduce function as {fun_name} available in this library')
                
            return fun
        
    @staticmethod
    def choose_mutate_fun(fun_name):
            if fun_name == '1+1':
                fun = mutations.mutate_one_plus_one
            elif fun_name == 'mu+lambda':
                fun = mutations.mutate_mu_plus_lambda
            else: 
                raise RuntimeError(f'there is no such mutate function as {fun_name} available in this library')
                
            return fun
    
    @staticmethod
    def choose_select_population_fun(fun_name):
            if fun_name == 'n_best':
                fun = selectpop.n_best_select
            elif fun_name == 'roulette_wheel':
                fun = selectpop.roulette_wheel_select
            else: 
                raise RuntimeError(f'there is no such select_population function as {fun_name} available in this library')
                
            return fun
        
    def train(self, starting_population, epochs,
              population_size=0, offspring_size=0, parents_per_child=2):
    
        history = []
        
        population = starting_population.copy()
        population = [np.array(x) for x in population]
        
        if ~population_size:
            population_size = len(population)
        if ~offspring_size:
            offspring_size = len(population)
        
        for i in range(epochs):
            
            offspring = []
            for i in range(offspring_size):
                try:
                    parents = self.select_parents(population, parents_per_child, self.cost_function)
                except TypeError:
                    parents = self.select_parents(population, parents_per_child)
                    
                try:
                    child = self.reproduce(parents, self.cost_function)
                except TypeError:
                    child = self.reproduce(parents)
                    
                try:
                    child = self.mutate(child, self.cost_function)
                except TypeError:
                    child = self.mutate(child)
                    
                if self.correct is not None:
                    try:
                        self.correct(child, self.cost_function)
                    except TypeError:
                        self.correct(child)
                        
                offspring.append(child)
                
            population.extend(offspring)
            
            try:
                population = self.select_population(population, population_size, self.cost_function)
            except TypeError:
                population = self.select_population(population, population_size)
        
            current_best_value = -sys.float_info.max
            for x in population:
                current_fun_value = self.cost_function(x)
                if current_fun_value > current_best_value:
                    self.current_best = x
                    current_best_value = current_fun_value
                    
            history.append(current_best_value)
                
        return history
    
class Memetic(Evolutional):
    
    def __init__(self, cost_function, cost_functions_gradient,
                 select_parents, reproduce, mutate,
                 select_population, correct=None):
        
        super().__init__(cost_function, select_parents, reproduce, mutate,
                         select_population, correct)
        
        self.cost_functions_gradient = cost_functions_gradient
        
        
    def train(self, starting_population, epochs,
              population_size=0, offspring_size=0, parents_per_child=2,
              local_optimization_iteration_number=10, memetic_learning_rate=0.1):
        
        history = []
        
        population = starting_population.copy()
        population = [np.array(x) for x in population]
        
        if ~population_size:
            population_size = len(population)
        if ~offspring_size:
            offspring_size = len(population)
        
        for i in range(epochs):
            
            offspring = []
            for i in range(offspring_size):
                try:
                    parents = self.select_parents(population, parents_per_child, self.cost_function)
                except TypeError:
                    parents = self.select_parents(population, parents_per_child)
                    
                try:
                    child = self.reproduce(parents, self.cost_function)
                except TypeError:
                    child = self.reproduce(parents)
                    
                try:
                    child = self.mutate(child, self.cost_function)
                except TypeError:
                    child = self.mutate(child)
                
                if self.correct is not None:
                    try:
                        self.correct(child, self.cost_function)
                    except TypeError:
                        self.correct(child)
                
                offspring.append(child)
                
            population.extend(offspring)
            memetic_population = []
            
            for speciman in population:
                speciman = self.local_optimization(speciman, local_optimization_iteration_number, memetic_learning_rate)
                memetic_population.append(speciman)
            
            population = [np.array(x) for x in memetic_population]
            
            try:
                population = self.select_population(population, population_size, self.cost_function)
            except TypeError:
                population = self.select_population(population, population_size)
        
            current_best_value = -sys.float_info.max
            for x in population:
                current_fun_value = self.cost_function(x)
                if current_fun_value > current_best_value:
                    self.current_best = x
                    current_best_value = current_fun_value
                    
            history.append(current_best_value)
                
        return history
    
    def local_optimization(self, item, iterations_number, learning_rate):
        
        for i in range(iterations_number):
            item += self.cost_functions_gradient(item) * learning_rate
            if self.correct is not None:
                try:
                    self.correct(item, self.cost_function)
                except TypeError:
                    self.correct(item)
        
        return item