#functions related to on-lattice simulation
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import pathlib
import pandas as pd
import time
from datetime import datetime


#this function randomly sample from neighborhood
def LEUP_uniform_onlattice(lattice,beta, bias,r, eta,neighbors_dict,memory,unique=False):
    """Perform synchronous updates using the Metropolis criterion."""
    new_lattice = np.copy(lattice)
    for i in range(np.shape(lattice)[0]):
        for j in range(np.shape(lattice)[0]):
            #neighbors = get_neighbors_lattice_site_with_radius(i, j, np.shape(lattice)[0], rs)
            neighbors=neighbors_dict[(i,j)]
            current_entropy = calculate_neighborhood_entropy_onlattice(lattice, neighbors)
            
            neighbors_values = np.array([lattice[x, y] for x, y in neighbors])
            if unique:
                neighbors_values=np.unique(neighbors_values)
        
            new_value=random.choice(neighbors_values)
            temp_lattice = lattice.copy()
            temp_lattice[i, j] = new_value
            new_entropy = calculate_neighborhood_entropy_onlattice(temp_lattice, neighbors)
            delta_s = new_entropy - current_entropy
            if  delta_s <0 or random.random() <= np.exp(-beta * delta_s):
                new_lattice[i, j] =new_value
    return new_lattice,memory

# this function sample bias way by integrating its memory with environmental cues  
def LEUP_memory_onlattice(lattice,beta, bias,r, eta,neighbors_dict,memory,unique=False):
    """Perform synchronous updates using the Metropolis criterion."""    
    new_lattice = np.copy(lattice)
    for i in range(np.shape(lattice)[0]):
        for j in range(np.shape(lattice)[0]):
            memory[str(i)+str(j)].append(lattice[i,j])
            memory_strength=bias
            neighbors=neighbors_dict[(i,j)]
            #neighbors =  get_neighbors_lattice_site_with_radius(i, j, np.shape(lattice)[0], r)
            current_entropy = calculate_neighborhood_entropy_onlattice(lattice, neighbors)
            
            neighbors_values = np.array([lattice[x, y] for x, y in neighbors])
            if unique:
                neighbors_values=np.unique(neighbors_values)
           
            memory_values=np.asarray(memory[str(i)+str(j)][min(len(memory[str(i)+str(j)]),-memory_strength):])
            integrated_arr=np.concatenate((neighbors_values,memory_values))
            new_value=random.choice(integrated_arr)

            temp_lattice = lattice.copy()
            temp_lattice[i, j] = new_value
            new_entropy = calculate_neighborhood_entropy_onlattice(temp_lattice, neighbors)
            delta_s = new_entropy - current_entropy
            if  delta_s <0 or random.random() <= np.exp(-beta * delta_s):
                new_lattice[i, j] = new_value
    return new_lattice,memory


# this is pure voter model
def pure_votermodel(lattice,beta, bias,r, eta,neighbors_dict,memory):
    """Perform synchronous updates using the Metropolis criterion."""
    new_lattice = np.copy(lattice)
    for i in range(np.shape(lattice)[0]):
        for j in range(np.shape(lattice)[0]):
            neighbors=neighbors_dict[(i,j)]
            nx, ny = random.choice(neighbors)
            new_lattice[i, j] = lattice[nx, ny]
    return new_lattice,memory


#this function compute neighborhood
def precompute_neighbors(size, radius):
    """
    Precomputes and returns a dictionary mapping each (i, j) index to the list of neighbor indices.
    """
    neighbors_dict = {}
    for i in range(size):
        for j in range(size):
            # using modulo arithmetic for periodic boundary conditions
            # create a list of all neighbors for cell (i, j)
            neighbors = [((i + dx) % size, (j + dy) % size)
                         for dx in range(-radius, radius + 1)
                         for dy in range(-radius, radius + 1)]
            neighbors_dict[(i, j)] = neighbors
    return neighbors_dict

#this function initialize lattice
def initialize_lattice(size):
    """Initialize the lattice with random votes."""
    return 2 * np.pi * np.random.rand(size, size)

#this function
def LEUP_von_misses_onlattice(lattice,beta, bias,r, eta,neighbors_dict,memory,unique=False):
    """Perform synchronous updates using the Metropolis criterion."""
    new_lattice = np.copy(lattice)
    
    for i in range(np.shape(lattice)[0]):
        for j in range(np.shape(lattice)[0]):
            #neighbors = get_neighbors_with_radius(i, j, size, radius)
            neighbors=neighbors_dict[(i,j)]
            current_entropy = calculate_neighborhood_entropy_onlattice(lattice, neighbors)
            neighbors_values = np.array([lattice[x, y] for x, y in neighbors])
            temp_lattice = lattice.copy()
            new_value = von_mises_weighted_choice(neighbors_values, bias)
            temp_lattice[i, j] =new_value
            new_entropy = calculate_neighborhood_entropy_onlattice(temp_lattice, neighbors)
            delta_s = new_entropy - current_entropy
            if delta_s <= 0 or random.random() < np.exp(-beta * delta_s):
                new_lattice[i, j] = new_value 
    return new_lattice,memory

# this function sense environmental decision in a biased way and then chose the angel based on entrpy minimization
def LEUP_von_misses_onlattice_biased_entropy(lattice,beta, bias,r, eta,neighbors_dict,memory,unique=False):
    """Perform synchronous updates using the Metropolis criterion."""
    new_lattice = np.copy(lattice)
    
    for i in range(np.shape(lattice)[0]):
        for j in range(np.shape(lattice)[0]):
            #neighbors = get_neighbors_with_radius(i, j, size, radius)
            neighbors=neighbors_dict[(i,j)]
            neighbors_values = np.array([lattice[x, y] for x, y in neighbors])

            # entropy for biased distirbution is computed here
            _,current_entropy=von_mises_weighted_choice(neighbors_values, bias)

            #propose a new decision
            temp_lattice = lattice.copy()
            new_value=random.choice(neighbors_values)
            temp_lattice[i, j] =new_value
            neighbors_values = np.array([temp_lattice[x, y] for x, y in neighbors])
            
            #entropy for the updated biased distribution is computed here
            aaa,new_entropy=von_mises_weighted_choice(neighbors_values, bias)

            # checking for acceptance of proposed decision
            delta_s = new_entropy - current_entropy
            if delta_s <= 0 or random.random() < np.exp(-beta * delta_s):
                new_lattice[i, j] = new_value 
    return new_lattice,memory

def calculate_neighborhood_entropy_onlattice(lattice, neighbors):
    """Calculate the entropy of a neighborhood."""
    values = [lattice[x, y] for x, y in neighbors]
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log(probabilities))

def calculate_neighborhood_entropy_onlattice2(neighbors_values):
    """Calculate the entropy of a neighborhood."""
    _, counts = np.unique(neighbors_values, return_counts=True)
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log(probabilities))