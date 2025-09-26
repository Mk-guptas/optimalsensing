import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pathlib
import pandas as pd
import time
import pickle
from datetime import datetime


# this function generat a random initial condition based on the number of indiviual and the dimension of the space    
def generate_initial_conditions_off_lattice(N, L):
    """Generate initial conditions with polar order less than 0.1."""
    while True:
        x = np.random.rand(N) * L
        y = np.random.rand(N) * L
        theta = 2 * np.pi * np.random.rand(N)
        polar_order = np.abs(np.sum(np.exp(1j * theta))) / N
        if polar_order < 0.1:
            return x, y, theta

# this function update the direction/decision in the off-lattice and also update the position of the individual if they have non-zero velocity
def LEUP_uniform_off_lattice(x,y,theta,beta,bias, r, eta,memory,v0,L,dt=0.05):
    """Simulate particle dynamics and return results."""
    order = 1  # 1 for Shannon entropy, >1 for Tsallis entropy

    # Update positions
    x, y = movement(x, y, theta, v0, dt, L)

    # Compute pairwise distances and neighborhood
    distance_matrix = np.sqrt((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2)
    neighbors = distance_matrix < r

    # Entropy before update
    S_before = np.array([calculate_entropy(theta[neighbors[j]], order) for j in range(len(x))])

    # Update angles
    updated_theta = theta.copy()
    for j in range(len(x)):
        if np.any(neighbors[j]):  # Ensure the particle has neighbors
            
            proposed_theta = random.choice(theta[neighbors[j]])
            temp_theta = theta.copy()
            temp_theta[j] = proposed_theta
            S_after = calculate_entropy(temp_theta[neighbors[j]], order)

            # Metropolis criterion
            if np.random.uniform() <= np.exp(-beta * (S_after - S_before[j])):
                updated_theta[j] = proposed_theta

    theta = np.copy(updated_theta)
    #plotting the image
    if False:
        plt.figure(figsize=(6,6))
        plt.quiver(x, y, np.cos(theta), np.sin(theta), angles='xy', scale_units='xy', scale=10)
        plt.savefig(output_path+'/image.jpg')
        plt.close()


    return x,y,theta

# this function updates postion and decision based on vicsek model in offlattice case
def vicsek_model(x,y,theta,beta,bias, r, eta,memory,v0,L,dt=0.05):

        # Update positions
        x, y = movement(x, y, theta, v0, dt, L)
        
        # Compute pairwise distances and neighborhood
        distance_matrix = np.sqrt((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2)
        neighbors = distance_matrix < r
    
        updated_theta = theta.copy()
        for j in range(len(x)):
            if np.any(neighbors[j]):  # Ensure the particle has neighbors
                neighbor_indices=neighbors[j] ;neighbors_values=theta[neighbor_indices];
                avg_theta_j=circular_mean(neighbors_values)
                
                # Continuous update: (avgTheta - currentTheta)
                diff = avg_theta_j - theta[j]
                
                # Noise term scaled by the number of neighbors
                num_neighbors = len(neighbor_indices) ;noise = eta *np.sqrt(num_neighbors)*random.uniform(-np.pi, np.pi)
                
                #langevin equation
                updated_theta[j] = theta[j] + dt * beta*diff  + noise

        theta=np.copy(updated_theta)
        return x,y,theta


# this function sample in a bised way from enivornmental option in decision making
def  LEUP_bias_off_lattice( x,y,theta,beta,bias, r, eta,memory,v0,L,dt=0.05):
    """Simulate particle dynamics and return results."""
    order = 1  # 1 for Shannon entropy, >1 for Tsallis entropy

    # Update positions
    x, y = movement(x, y, theta, v0, dt, L)

    # updating decisions
    
    # Compute pairwise distances and neighborhood
    distance_matrix = np.sqrt((x[:, None] - x)**2 + (y[:, None] - y)**2)
    neighbors = distance_matrix < r

    # Entropy before update
    S_before = np.array([calculate_entropy(theta[neighbors[j]], order) for j in range(N)])

    # Update angles using Von Mises weighted selection
    updated_theta = theta.copy()
    for agent_idx in range(len(x)):
        if np.any(neighbors[agent_idx]):  # Ensure the particle has neighbors
            
            proposed_theta = von_mises_weighted_choice(theta[neighbors[agent_idx]],bias )
            temp_theta = theta.copy()
            temp_theta[agent_idx] = proposed_theta
            S_after = calculate_entropy(temp_theta[neighbors[agent_idx]], order)

            # Metropolis criterion
            if np.random.uniform() <= np.exp(-beta * (S_after - S_before[agent_idx])):
                updated_theta[agent_idx] = proposed_theta

    theta = np.copy(updated_theta)
    # Store results
    return x,y,theta



# this function sample in a bised way from enivornmental option in decision making
def  LEUP_von_misses_offlattice_biased_entropy(x,y,theta,beta,bias, r, eta,memory,v0,L,dt=0.05):
    """Simulate particle dynamics and return results."""
    order = 1  # 1 for Shannon entropy, >1 for Tsallis entropy
    # Update positions
    x, y = movement(x, y, theta, v0, dt, L)

    # Compute pairwise distances and neighborhood
    distance_matrix = np.sqrt((x[:, None] - x)**2 + (y[:, None] - y)**2)
    neighbors = distance_matrix < r

    # Entropy before update
    S_before=np.array([von_mises_weighted_choice(theta[neighbors[j]], bias)[1] for j in range(len(x))])
    
    # Update angles using Von Mises sensing distribution
    updated_theta = theta.copy()
    for agent_idx in range(len(x)):
        if np.any(neighbors[agent_idx]):  # Ensure the particle has neighbors
            
            proposed_theta = random.choice(theta[neighbors[agent_idx]])
            temp_theta = theta.copy()
            temp_theta[agent_idx] = proposed_theta
            S_after = von_mises_weighted_choice(temp_theta[neighbors[agent_idx]], bias)[1]

            # Metropolis criterion
            if np.random.uniform() <= np.exp(-beta * (S_after - S_before[agent_idx])):
                updated_theta[agent_idx] = proposed_theta
    theta = np.copy(updated_theta)
    # Store results
    return x,y,theta


# this function integrate memory with enivornmental option in decision making
def  LEUP_memory_off_lattice( x,y,theta,beta,bias, r, eta,memory,v0,L,dt=0.05):
    """Simulate particle dynamics and return results."""
    order = 1  # 1 for Shannon entropy, >1 for Tsallis entropy

    # Update positions
    x, y = movement(x, y, theta, v0, dt, L)

    # updating decisions
    
    # Compute pairwise distances and neighborhood
    distance_matrix = np.sqrt((x[:, None] - x)**2 + (y[:, None] - y)**2)
    neighbors = distance_matrix < r

    # Entropy before update
    S_before = np.array([calculate_entropy(theta[neighbors[j]], order) for j in range(N)])

    # Update angles using Von Mises weighted selection
    updated_theta = theta.copy()
    for agent_idx in range(len(x)):
        if np.any(neighbors[agent_idx]):  # Ensure the particle has neighbors
            memory[str(agent_idx)].append(theta[agent_idx])
            memory_strength=bias
            memory_values=np.asarray(memory[str(agent_idx)][min(len(memory[str(agent_idx)]),-memory_strength):])
            integrated_arr=np.concatenate((theta[neighbors[agent_idx]],memory_values))
            new_value=random.choice(integrated_arr)
            
            temp_theta = theta.copy()
            temp_theta[agent_idx] = new_value
            S_after = calculate_entropy(temp_theta[neighbors[agent_idx]], order)

            # Metropolis criterion
            if np.random.uniform() <= np.exp(-beta * (S_after - S_before[agent_idx])):
                updated_theta[agent_idx] = new_value

    theta = np.copy(updated_theta)
    # Store results
    return x,y,theta



# this function calculate entropy
def calculate_entropy(labels, order, base=None):
    """Calculate Shannon or Tsallis entropy."""
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = np.e if base is None else base
    return -(vc * np.log(vc) / np.log(base)).sum()

#this function does the movement of each particle
def movement(x, y, theta, v0, dt, L):
    """Update positions of particles with periodic boundary conditions."""
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x = (x + vx * dt) % L
    y = (y + vy * dt) % L
    return x, y

#this function computes the circular mean for vicsek model
def circular_mean(angles):
    """Compute the circular mean of a list of angles (in radians)."""
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)


#this function assign weigh to each decision according to vonmis
def von_mises_weighted_choice(neighbors_values, kappa):
    """Select a value from neighbors based on Von Mises distribution."""
    mean_value = circular_mean(neighbors_values)
    unique_decisions, counts = np.unique(neighbors_values, return_counts=True)
    p_occurance = counts / np.sum(counts)

    weights = np.exp(kappa * np.cos(unique_decisions - mean_value))
    probabilities =   p_occurance*weights / np.sum(p_occurance*weights)
    
    entropy=-np.sum(probabilities * np.log(probabilities))
    return np.random.choice(unique_decisions, p=probabilities),entropy
    