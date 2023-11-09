import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math 
import sys
from datetime import datetime
from curses import wrapper
import pandas as pd

pop_size = 20
selection = 0.2
layers = 3
neurons_per_layer = 10
total_neurons = layers * neurons_per_layer
mutation = 1./ total_neurons
generations = 100
num_parents = int(pop_size*selection)
num_children = pop_size - num_parents

num_targets = 20
area_x = 500
area_y = 500
simulation_interations = 100

class Target:
   def __init__(self, x, y):
      self.x = x
      self.y = y

   def get_position(self):
      return self.x, self.y
   
   def set_position(self, x, y):
      self.x = x
      self.y = y

class Brain:
   def __init__(self, neurons_per_layer):
      self.x = 0
      self.y = 0
      self.rotation = 0
      self.lookat_x = 0
      self.lookat_y = 0

      #our model has 1 input being rotation to target.
      #It has 2 outputs representing the movement required to each tank track
      layer_a = tf.keras.layers.Dense(neurons_per_layer, activation='relu', input_shape=[1])
      layer_b = tf.keras.layers.Dense(neurons_per_layer, activation='relu')
      layer_c = tf.keras.layers.Dense(neurons_per_layer, activation='sigmoid')
      output_layer = tf.keras.layers.Dense(2)

      self.model = tf.keras.models.Sequential()
      self.model.add(layer_a)
      self.model.add(layer_b)
      self.model.add(layer_c)
      self.model.add(output_layer)
      self.model.build()

   def set_weight(self, kernal, bias, layer_idx):
      self.model.get_layer(index=layer_idx).set_weights([kernal, bias])

   def get_kernal(self, layer_idx):
      kernal, bias = self.model.get_layer(index=layer_idx).get_weights()
      return kernal
   
   def get_bias(self, layer_idx):
      kernal, bias = self.model.get_layer(index=layer_idx).get_weights()
      return bias
   
   def get_position(self):
      return self.x, self.y
   
   def set_position(self, x, y):
      self.x = x
      self.y = y

   def get_rotation(self):
      return self.rotation
   
   def set_rotation(self, rotation):
      self.rotation = rotation

   def get_lookat(self):
      return self.lookat_x, self.lookat_y
   
   def set_lookat(self, x, y):
      self.lookat_x = x
      self.lookat_y = y

   def predict_movement(self, rotation_to_target):
      output = self.model.predict_on_batch(np.array([rotation_to_target]))
      return output[0]

################Start model learning Logic################
def gather_flattened_kernal_matricies(brains, num_neurons, layer_idx):
   flat_kernals = np.ones(shape=[len(brains), num_neurons])
   for brain_idx in range(len(brains)):
      flat_kernals[brain_idx] = brains[brain_idx].get_kernal(layer_idx).flatten()

   return flat_kernals

def gather_bias(brains, bias_len, layer_idx):
   flat_bias = np.ones(shape=[len(brains), bias_len])
   for brain_idx in range(len(brains)):
      flat_bias[brain_idx] = brains[brain_idx].get_bias(layer_idx)

   return flat_bias

def generate_new_population_values(population_values_matrix, top_indices, values_len, crossover_mat, rand_parent1_ix, rand_parent2_ix, mutation_prob_mat, mutation_values):
   population_sorted = tf.gather(population_values_matrix, top_indices)
   parents = tf.slice(population_sorted, [0, 0], [num_parents, values_len])

   #gather parents by shuffled indices
   rand_parent1 = tf.gather(parents, rand_parent1_ix)
   rand_parent2 = tf.gather(parents, rand_parent2_ix)
   rand_parent1_sel = tf.math.multiply(rand_parent1, crossover_mat)

   rand_parent2_sel = tf.math.multiply(rand_parent2, tf.math.subtract(np.float64(1.), crossover_mat))
   children_after_sel = tf.math.add(rand_parent1_sel, rand_parent2_sel)

   mutation_values[mutation_prob_mat >= mutation] = 0
   mutated_children = tf.math.add(children_after_sel, mutation_values)
   #combine children and parents into new population
   population_brains = tf.concat([parents, mutated_children], 0)
   return population_brains

def generate_new_dense_layer(brains, top_indices, num_inputs, num_outputs, layer_idx):
   num_of_neurons = num_inputs * num_outputs
   bias_count = neurons_per_layer

   #initialise values
   crossover_mat = np.ones(shape=[num_children, num_of_neurons], dtype='float64')
   crossover_point = np.random.choice(np.arange(1, num_of_neurons-1, step=1), num_children)
   for pop_ix in range(num_children):
      crossover_mat[pop_ix,0:crossover_point[pop_ix]]=0.

   bias_crossover_mat = np.ones(shape=[num_children, num_outputs], dtype='float64')

   #leave output values as is to avoid logic issues
   if num_outputs > 2:
      bias_crossover_point = np.random.choice(np.arange(1, num_outputs-1, step=1), num_children)
      for pop_ix in range(num_children):
         bias_crossover_mat[pop_ix,0:bias_crossover_point[pop_ix]]=0.

   #flatten the kernals to make them easier to work with
   pop_flattened_kernal = gather_flattened_kernal_matricies(brains, num_of_neurons, layer_idx)
   pop_bias = gather_bias(brains, num_outputs, layer_idx)

   #gather indicies to shuffle parents
   rand_parent1_ix = np.random.choice(num_parents, num_children)
   rand_parent2_ix = np.random.choice(num_parents, num_children)

   #generate mutation probability matrices
   mutation_prob_mat = np.random.uniform(size=[num_children, num_of_neurons])
   mutation_values = np.random.normal(size=[num_children, num_of_neurons])

   bias_mutation_prob_mat = np.random.uniform(size=[num_children, num_outputs])
   bias_mutation_values = np.random.normal(size=[num_children, num_outputs])

   new_pop_kernals = generate_new_population_values(pop_flattened_kernal, top_indices, num_of_neurons, crossover_mat, rand_parent1_ix, rand_parent2_ix, mutation_prob_mat, mutation_values)
   new_pop_bias = generate_new_population_values(pop_bias, top_indices, num_outputs, bias_crossover_mat, rand_parent1_ix, rand_parent2_ix, bias_mutation_prob_mat, bias_mutation_values)

   #copy over new generation values
   for brain_idx in range(len(brains)):
      brains[brain_idx].set_weight(np.reshape(new_pop_kernals[brain_idx], (-1, num_outputs)), new_pop_bias[brain_idx], layer_idx)

   return brains
################End model learning Logic################

################Start Game Logic################
def direction_sign(from_x, from_y, to_x, to_y):
   if from_y*to_x > from_x*to_y:
      return 1
   else:
      return -1

def dot_product(x1, y1, x2, y2):
   return x1*x2 + y1*y2

def normalize(x, y):
   magSqr = (x*x) + (y*y)
   if magSqr > 0.:
      oneOverMag = 1./math.sqrt(magSqr)
      x *= oneOverMag
      y *= oneOverMag

   return x, y

def direction_vector(from_x, from_y, to_x, to_y):
   x = from_x - to_x
   y = from_y - to_y
   return x, y

def get_rotation_to_target(lookat_x, lookat_y, from_x, from_y, to_x, to_y):
   x,y = direction_vector(from_x, from_y, to_x, to_y)
   x,y = normalize(x,y)
   dot = dot_product(lookat_x, lookat_y, x, y)
   sign = direction_sign(lookat_x, lookat_y, x, y)

   return dot * sign

def get_distance(from_x, from_y, to_x, to_y):
   x = from_x-to_x
   y = from_y-to_y
   return math.sqrt(x*x + y*y)

def generate_random_position(area_x, area_y):
   x = np.random.randint(area_x)
   y = np.random.randint(area_y)
   return x,y

def generate_targets(num_targets, area_x, area_y):
   targets = []
   for i in range(num_targets):
      x,y = generate_random_position(area_x, area_y)
      targets.append(Target(x, y))
   return targets

def assign_random_positions(population):
   for p in population:
      x = np.random.randint(area_x)
      y = np.random.randint(area_y)
      p.set_position(x, y)
      p.set_rotation(0.)

def find_closest_target(targets, x, y):
   closest = sys.float_info.max
   idx = 0
   for t in range(len(targets)):
      tx, ty = targets[t].get_position()
      dist = get_distance(x, y, tx, ty)
      if dist < closest:
         closest = dist
         idx = t
   
   return idx

def set_new_tank_position(tank, t1, t2, max_turn_rate, max_speed, dt, area_x, area_y):
   #calculate the steering force
   rotation_force = (t1 - t2) * max_turn_rate

	#clamp the rotation
   if rotation_force > max_turn_rate:
      rotation_force = max_turn_rate
   elif rotation_force < -max_turn_rate:
      rotation_force = -max_turn_rate

   #set rotation to target
   rotation = tank.get_rotation()
   rotation += rotation_force

   speed = (t1 + t2) * max_speed

   #clamp the speed
   if speed > max_speed:
      speed = max_speed

   #calculate the direction
   lookat_x = -math.sin(rotation)
   lookat_y = math.cos(rotation)

	#update position
   pos_x, pos_y = tank.get_position()
   pos_x += lookat_x*speed * dt
   pos_y += lookat_y*speed * dt

   #wrap around window limits
   if (pos_x > area_x):
      pos_x = 0.
   if (pos_x < 0):
      pos_x = area_x
   if (pos_y > area_y):
      pos_y = 0.
   if (pos_y < 0):
      pos_y = area_y

   #set new tank values
   tank.set_position(pos_x, pos_y)
   tank.set_lookat(lookat_x, lookat_y)
   tank.set_rotation(rotation)

def render(screen, population, targets, generation):
   screen.clear()
   screen.addstr(0,0, "Generation " + str(generation))
   for tank_idx in range(len(population)):
      tank_x, tank_y = population[tank_idx].get_position()
      screen.addch(int(tank_y / 10), int(tank_x / 5), "X")

   for target_idx in range(len(targets)):
      tx, ty = targets[target_idx].get_position()
      screen.addch(int(ty / 10), int(tx / 5), "O")

   screen.refresh()

def run_simulation(screen, population, num_targets, area_x, area_y, generation):
   population_fitness = np.zeros(len(population))

   targets = generate_targets(num_targets, area_x, area_y)
   assign_random_positions(population)

   last_update_time = datetime.now()

   #simulation values tweaked to what feels correct
   max_turn_rate = 0.3
   max_speed = 10
   dt = 1
   hit_distance = 10

   iteration = 0

   #each simulation run has a fixed number of iterations
   for sim in range(simulation_interations):
      now_time = datetime.now()
      dt = (now_time - last_update_time).total_seconds()
      last_update_time = now_time

      iteration+=1
      tank_x, tank_y = population[0].get_position()

      #move tanks
      for tank_idx, tank in enumerate(population):
         tank_x, tank_y = tank.get_position()
         idx = find_closest_target(targets, tank_x,tank_y)
         targ_x, targ_y = targets[idx].get_position()
         lookat_x, lookat_y = tank.get_lookat()
         rotation = get_rotation_to_target(lookat_x, lookat_y, tank_x, tank_y, targ_x, targ_y)

         #feed rotation to target into our model to calculate movement values
         #The value returned is the force required to each track on the tank
         t1, t2 = tank.predict_movement(rotation)
         set_new_tank_position(tank, t1, t2, max_turn_rate, max_speed, dt, area_x, area_y)

         #check if targets reached
         tank_x, tank_y = tank.get_position()
         dist_to_target = get_distance(tank_x, tank_y, targ_x, targ_y)
         if dist_to_target <= hit_distance:
            #increment this tanks fitness then spwan another target
            population_fitness[tank_idx] += 1
            x,y = generate_random_position(area_x, area_y)
            targets[idx].set_position(x, y)

      render(screen, population, targets, generation)

   return population_fitness
################End Game Logic################

def main(screen):
   screen.resize(250, 250)
   brains = []
   
   colnames = []
   for i in range(pop_size):
      colnames.append(str(i))
   training_report = pd.DataFrame(columns=colnames)
   
   #create the initial set of models
   for i in range(pop_size):
      brains.append(Brain(neurons_per_layer))

   #the fittest individuals are selected after each simulation run
   # to create the next generation of tanks 
   for i in range(generations):
      population_fitness = run_simulation(screen, brains, num_targets, area_x, area_y, i)
      top_vals, top_ind = tf.nn.top_k(population_fitness, k=pop_size)

      training_report.loc[i] = population_fitness
      if (i % 10) == 0:
         training_report.to_csv(str(i)+".csv", index=False)

      #generate new layers using the fittest individuals
      brains = generate_new_dense_layer(brains, top_ind, 1, neurons_per_layer, 0)
      brains = generate_new_dense_layer(brains, top_ind, neurons_per_layer, neurons_per_layer, 1)
      brains = generate_new_dense_layer(brains, top_ind, neurons_per_layer, neurons_per_layer, 2)
      brains = generate_new_dense_layer(brains, top_ind, neurons_per_layer, 2, 3)

wrapper(main)
