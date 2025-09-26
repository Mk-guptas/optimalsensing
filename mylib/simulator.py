#functions related to simulator
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pathlib
import pandas as pd
import time
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def simulator_offlattice(model_type,model_parameters,output_path):
    
    no_of_trajectory=model_parameters.get('no_of_trajectory',1);
    N=model_parameters.get('No_of_particle',250);
    simulation_steps=model_parameters.get('simulation_steps',100);
    beta_list=model_parameters.get('beta_list',[0]);
    R_list=model_parameters.get('R_list',[1]);
    speed_list=model_parameters.get('speed_list',[0.1]);
    
    L_list=model_parameters.get('L_list',[10]);
    bias_param_list=model_parameters.get('bias_param_list',[0]);
    eta=model_parameters.get('eta',0);
    polar_order_threshold=model_parameters.get(' polar_order_threshold',0.9);

    information_dictionary={'no_of_trajectory':no_of_trajectory,'model_type':model_type,'noise_scale':eta,'simulation_steps':simulation_steps,'polar_order_threshold':polar_order_threshold,'no_of_particle':N, \
                           'speed_list':speed_list,'R_list':R_list,'beta_list':beta_list,'L_list':L_list,'bias_param_list':bias_param_list}
    
    with open(os.path.join(output_path, 'information_dictionary.pkl'), 'wb') as f:
        pickle.dump(information_dictionary, f)


    for trajectory in range(no_of_trajectory):
    
        ## Output arrays
        ensemble_polar_order = np.zeros((len(L_list), len(speed_list), len(bias_param_list),len(beta_list), len(R_list), simulation_steps),dtype=np.float16)
        ensemble_angle_values = np.zeros((len(L_list), len(speed_list),len(bias_param_list), len(beta_list), len(R_list) ,simulation_steps,N),dtype=np.float16)
        time_to_reach_0_9 =  np.zeros((len(L_list), len(speed_list),len(bias_param_list), len(beta_list), len(R_list)),dtype=np.float16)*np.nan
        
        for L_index, L in enumerate(L_list):
            initial_conditions = generate_initial_conditions_off_lattice(N, L)
            for v_index, v in enumerate(speed_list):
                for bias_index, bias in enumerate(bias_param_list):
                    for beta_index, beta in enumerate(beta_list):
                        for r_index, r in enumerate(R_list):
                            # Use the same initial conditions
                            x, y, theta = initial_conditions ; time_to_0_9=None
                            memory={str(agent_idx):[] for agent_idx in range(N)}
                            for time_idx in range(simulation_steps):
                                x,y,theta= globals()[model_type](x,y,theta,beta,bias, r, eta,memory,v,L)
                                polar_order = np.abs(np.sum(np.exp(1j * theta))) / N
                                ensemble_polar_order[L_index, v_index,bias_index, beta_index, r_index,time_idx] = polar_order
                                ensemble_angle_values[L_index, v_index,bias_index, beta_index, r_index,time_idx] = theta.copy()
                                
                             # Early stopping condition
                                if polar_order >= polar_order_threshold:
                                    ensemble_polar_order[L_index, v_index, bias_index,beta_index, r_index][time_idx+1:] = polar_order
                                    ensemble_angle_values[L_index, v_index,bias_index, beta_index, r_index][time_idx+1:] = theta.copy()
                                    time_to_reach_0_9[L_index, v_index, bias_index,beta_index, r_index]=time_idx 
                                    break
                            print(f" model_type {model_type} size of space {L}, velocity {v}, bias {bias} no of particle {N}")
                            print(f"Trajectory {trajectory}, Beta {beta}, R {r}, T{time_to_0_9}")       

        # Save data
        np.save(os.path.join(output_path, str(trajectory)+'_ensemble_polar_order.npy'), ensemble_polar_order)
        np.save(os.path.join(output_path, str(trajectory)+'_ensemble_angle_values.npy'), ensemble_angle_values)
        np.save(os.path.join(output_path, str(trajectory)+'_time_to_reach_0_9.npy'), time_to_reach_0_9)


# this function simulate onlattice
def simulator_onlattice(model_type,model_parameters,output_path):
    
    no_of_trajectory=model_parameters.get('no_of_trajectory',1);
    N=model_parameters.get('No_of_particle',[250]);
    simulation_steps=model_parameters.get('simulation_steps',100);
    beta_list=model_parameters.get('beta_list',[0]);
    R_list=model_parameters.get('R_list',[1]);
    speed_list=model_parameters.get('speed_list',[0]);
    
    L_list=model_parameters.get('L_list',[15]);
    bias_param_list=model_parameters.get('bias_param_list',[0]);
    eta=model_parameters.get('eta',0);
    polar_order_threshold=model_parameters.get(' polar_order_threshold',0.9);

    information_dictionary={'no_of_trajectory':no_of_trajectory,'model_type':model_type,'noise_scale':eta,'simulation_steps':simulation_steps,'polar_order_threshold':polar_order_threshold,'no_of_particle':L_list[0]**2,\
                           'speed_list':speed_list,'R_list':R_list,'beta_list':beta_list,'L_list':L_list,'bias_param_list':bias_param_list}
    
    with open(os.path.join(output_path, 'information_dictionary.pkl'), 'wb') as f:
        pickle.dump(information_dictionary, f)


    for trajectory in range(no_of_trajectory):
    
        ## Output arrays
        ensemble_polar_order = np.zeros((len(L_list), len(speed_list), len(bias_param_list),len(beta_list), len(R_list), simulation_steps),dtype=np.float16)
        ensemble_angle_values = np.zeros((len(L_list), len(speed_list),len(bias_param_list), len(beta_list), len(R_list) , simulation_steps,L_list[0],L_list[0]),dtype=np.float16)
        time_to_reach_0_9 =  np.zeros((len(L_list), len(speed_list),len(bias_param_list), len(beta_list), len(R_list)),dtype=np.float16)*np.nan

    
        for L_index, L in enumerate(L_list):
            initial_conditions=initialize_lattice(L)

            
            for v_index, v in enumerate(speed_list):
                for bias_index, bias in enumerate(bias_param_list):
                    for beta_index, beta in enumerate(beta_list):
                        for r_index, r in enumerate(R_list):
                            # Use the same initial conditions
                            lattice= np.copy(initial_conditions)  ;time_to_0_9=None ;
                            neighbors_dict=precompute_neighbors(np.shape(lattice)[0], r)
                            memory={str(row)+str(col):[] for row in range(np.shape(lattice)[0]) for col in range(np.shape(lattice)[0])}
                            
                            for time_idx in range(simulation_steps):
                               
                                lattice,memory= globals()[model_type](lattice,beta, bias,r, eta,neighbors_dict,memory)
                                theta = lattice.flatten()
                                polar_order = np.abs(np.sum(np.exp(1j * theta))) / len(theta)
                                ensemble_polar_order[L_index, v_index,bias_index, beta_index, r_index,time_idx] = polar_order
                                ensemble_angle_values[L_index, v_index,bias_index, beta_index, r_index,time_idx] = lattice.copy()   #saving lattice
                                
                             # Early stopping condition
                                if polar_order >= polar_order_threshold:
                                    time_to_0_9 = time_idx 
                                    ensemble_polar_order[L_index, v_index, bias_index,beta_index, r_index][time_idx+1:] = polar_order
                                    ensemble_angle_values[L_index, v_index,bias_index, beta_index, r_index][time_idx+1:] = lattice.copy()
                                    time_to_reach_0_9[L_index, v_index, bias_index,beta_index, r_index]=time_to_0_9
                                    break
                            print(f" model_type {model_type} size of space {L}, velocity {v}, bias {bias} no of particle {N}")
                            print(f"Trajectory {trajectory}, Beta {beta}, R {r}, T{time_to_0_9}")       

        # Save data
        np.save(os.path.join(output_path, str(trajectory)+'_ensemble_polar_order.npy'), ensemble_polar_order)
        np.save(os.path.join(output_path, str(trajectory)+'_ensemble_angle_values.npy'), ensemble_angle_values)
        np.save(os.path.join(output_path, str(trajectory)+'_time_to_reach_0_9.npy'), time_to_reach_0_9)
