#plotter functions
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
from matplotlib.colors import LogNorm

# Function to plot polar order
def polar_order_plotter(output_path):

    #loading informtation regarding the simulation
    with open(os.path.join(output_path, 'information_dictionary.pkl'), 'rb') as f:
        information_dictionary = pickle.load(f)
        
    no_of_trajectory=information_dictionary['no_of_trajectory']
    simulation_steps=information_dictionary['simulation_steps']
    no_of_particle=information_dictionary['no_of_particle']
    speed_list=information_dictionary['speed_list']
    R_list=information_dictionary['R_list']
    beta_list=information_dictionary['beta_list']
    L_list=information_dictionary['L_list']
    bias_param_list=information_dictionary['bias_param_list']

    #loading simulated data
    ensemble_polar_order=np.asarray([np.load(output_path+'/'+str(i)+'_ensemble_polar_order.npy') for i in range(no_of_trajectory)])
    time_to_reach_0_9=np.asarray([np.load(output_path+'/'+str(i)+'_time_to_reach_0_9.npy') for i in range(no_of_trajectory)])

    

    #checking the loaded information if it mathces
    print(f"no_of_trajectory{no_of_trajectory}, simulation_steps {simulation_steps}  No_of_particle {no_of_particle}, speed_list {speed_list}, L_list {L_list }")
    print(f"R_list{R_list},beta_list{beta_list}, bias_param_list{bias_param_list}, shape {np.shape(ensemble_polar_order)}")

    # averaging polar order
    avg_ensemble_polar_order=np.average(ensemble_polar_order,axis=0)

    print(f" avg_polar_order{np.shape(avg_ensemble_polar_order)}")
    
    for bias_index, bias in enumerate(bias_param_list):

        fig, axes = plt.subplots(2, 3, figsize=(15, 6),sharey=True,sharex=True)
        #if len(beta_list) == 1:
            #axes = [axes]  # Ensure axes is iterable if there's only one beta value
        for beta_index, beta in enumerate(beta_list):
            for r_index, r in enumerate(R_list):
                axes[beta_index//3,beta_index%3].plot(np.arange(0,simulation_steps),avg_ensemble_polar_order[0,0,bias_index,beta_index,r_index], label=f'R={r}')
    

            #styling
            axes[beta_index//3,beta_index%3].legend(frameon=False, fontsize=7)
            axes[beta_index//3,beta_index%3].set_title(fr'$\beta$={beta}')
            axes[beta_index//3,beta_index%3].plot(np.arange(0,simulation_steps),np.ones(simulation_steps)*0.9,linestyle='--',color='black')
           
            axes[beta_index//3,beta_index%3].tick_params(direction='out',labelsize=15,  size=5, width=2,  pad=5 )        
        
            axes[beta_index//3,beta_index%3].spines['top'].set_visible(False)  # Hide top border
            axes[beta_index//3,beta_index%3].spines['right'].set_visible(False)  # Hide right border
            axes[beta_index//3,beta_index%3].spines['left'].set_linewidth(2)  # Make left spine thicker
            axes[beta_index//3,beta_index%3].spines['bottom'].set_linewidth(2)  # Make bottom spine thicker
        
        #axes[beta_index//3,beta_index%3].set_xlabel("Time-step", fontsize=10 ,fontweight='bold')  # Axis label styling
        fig.supylabel("Polar order", fontsize=13,fontweight='bold')
        fig.supxlabel("Time-step", fontsize=13,fontweight='bold')
        #fig.delaxes(axes[1, 2])
        #fig.supylabel("Shared Y-axis", fontsize=14)
        #axes[-1].set_xlabel('Time Step')
        #axes[beta_index//3,beta_index%3].legend()
        plt.suptitle('Average Polar Order Over Time for Different R Values and Beta')
        plt.subplots_adjust(hspace=0.2,wspace=0.2)
        plt.show()

def time_to_reach_consensus(output_path,bias_idx=0):

    #loading informtation regarding the simulation
    with open(os.path.join(output_path, 'information_dictionary.pkl'), 'rb') as f:
        information_dictionary = pickle.load(f)
        
    no_of_trajectory=information_dictionary['no_of_trajectory']
    simulation_steps=information_dictionary['simulation_steps']
    no_of_particle=information_dictionary['no_of_particle']
    speed_list=information_dictionary['speed_list']
    R_list=information_dictionary['R_list']
    beta_list=information_dictionary['beta_list']
    L_list=information_dictionary['L_list']
    bias_param_list=information_dictionary['bias_param_list']

    #loading simulated data
    ensemble_polar_order=np.asarray([np.load(output_path+'/'+str(i)+'_ensemble_polar_order.npy') for i in range(no_of_trajectory)])
    all_time_to_reach_0_9=np.asarray([np.load(output_path+'/'+str(i)+'_time_to_reach_0_9.npy') for i in range(no_of_trajectory)])

    #checking the loaded information if it mathces
    print(f"no_of_trajectory{no_of_trajectory}, simulation_steps {simulation_steps}  No_of_particle {no_of_particle}, speed_list {speed_list}, L_list {L_list }")
    print(f"R_list{R_list},beta_list{beta_list}, bias_param_list{bias_param_list}, shape {np.shape(ensemble_polar_order)}")


    # averaging polar order
    avg_ensemble_polar_order=np.average(ensemble_polar_order,axis=0)

    # averaging time to consensus
    all_time_to_reach_0_9 = np.array(all_time_to_reach_0_9, dtype=float)
    avg_time_to_0_9 = np.nanmean(all_time_to_reach_0_9, axis=0)  # Assuming avg_time_to_0_9 is 3D: (X, Y, beta_index)
    
    # Define X (Radius R) and Y (Beta values)
    R_list = np.array(R_list[:])  # X-axis: Radius
    beta_list = np.array(beta_list[:])  # Y-axis: Beta values
    
    # Extract relevant data for heatmap (assuming axis 0,0 is fixed)
    heatmap_data = avg_time_to_0_9[0, 0,bias_idx]  # Shape should be (Beta, R)
    heatmap_data=np.where(avg_ensemble_polar_order[0,0,bias_idx,:,:,-1]>=0.9,heatmap_data,np.nan)
    heatmap_data=heatmap_data[:,:]


    # Create the figure
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Normalize data to enhance small differences
    norm = mcolors.Normalize(vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data))
    norm =LogNorm(vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data)) 
    # Create the heatmap
    cmap = plt.cm.viridis  # Colormap choice (you can try 'plasma', 'magma', or 'cividis')
    c = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, norm=norm, origin='lower')
    
    # Set axis labels and title
    ax.set_xticks(np.arange(len(R_list)))  ; ax.set_xticklabels(R_list)
    ax.set_yticks(np.arange(len(beta_list))) ; ax.set_yticklabels(beta_list)
   
    #ax.set_xlabel('R', fontsize=12,fontweight='bold')
    #ax.set_ylabel(fr'$\beta$', fontsize=12,fontweight='bold')
    ax.set_title('Heatmap: Time-steps', fontsize=14)
    #styling
    ax.tick_params( direction='out',   size=5, width=2,  pad=5   )          
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    #cbar.set_label('Time steps', fontsize=12)
    
    # Show plot
    plt.tight_layout()
    plt.savefig(output_path+"/phasespace_off.svg", format="svg",bbox_inches='tight')
    plt.show()

# normlaized time consensus
def normalized_time_consensus(output_path, path, plotting_param, plotting_idx=0):

    #loading informtation regarding the simulation
    with open(os.path.join(output_path, 'information_dictionary.pkl'), 'rb') as f:
        information_dictionary = pickle.load(f)
        
    no_of_trajectory=information_dictionary['no_of_trajectory']
    simulation_steps=information_dictionary['simulation_steps']
    no_of_particle=information_dictionary['no_of_particle']
    speed_list=information_dictionary['speed_list']
    R_list=information_dictionary['R_list']
    beta_list=information_dictionary['beta_list']
    L_list=information_dictionary['L_list']
    bias_param_list=information_dictionary['bias_param_list']

    #loading simulated data
    ensemble_polar_order=np.asarray([np.load(output_path+'/'+str(i)+'_ensemble_polar_order.npy') for i in range(no_of_trajectory)])
    all_time_to_reach_0_9=np.asarray([np.load(output_path+'/'+str(i)+'_time_to_reach_0_9.npy') for i in range(no_of_trajectory)])

    #checking the loaded information if it mathces
    print(f"no_of_trajectory{no_of_trajectory}, simulation_steps {simulation_steps}  No_of_particle {no_of_particle}, speed_list {speed_list}, L_list {L_list }")
    print(f"R_list{R_list},beta_list{beta_list}, bias_param_list{bias_param_list}, shape {np.shape(ensemble_polar_order)}")

    
    # Convert data to float and compute averages and standard deviations over trajectories
    all_time_to_reach_0_9 = np.array(all_time_to_reach_0_9, dtype=float)
    avg_time_to_0_9 = np.nanmean(all_time_to_reach_0_9, axis=0)
    std_time_to_0_9 = np.nanstd(all_time_to_reach_0_9, axis=0)
    
    # Create a figure with two subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    
    if plotting_param == 'beta':
        for beta_index, beta in enumerate(beta_list):
            # Extract the slice corresponding to the current beta and chosen plotting index.
            avg_values = avg_time_to_0_9[0, 0, plotting_idx, beta_index] ;std_values = std_time_to_0_9[0, 0, plotting_idx, beta_index]
            # For the normalized plot, divide the values and error by the maximum value in this slice.
            max_val = np.nanmax(avg_values) ; normalized_avg = avg_values / max_val  ;normalized_std = std_values / max_val
           
            # Plot normalized data in the left subplot (ax[0])
            ax[0].errorbar(R_list, normalized_avg, yerr=normalized_std, marker='o',linestyle='--', label=fr'$\beta$={beta}')
            
            # Plot unnormalized data in the right subplot (ax[1])
            ax[1].errorbar(R_list, avg_values, yerr=std_values, marker='o', linestyle='--', label=fr'$\beta$={beta}')
    
    elif plotting_param == 'bias':
        for bias_idx, bias in enumerate(bias_param_list):
            # Extract the data slice for the current bias parameter.
            #if bias==1 or bias==10 or bias==20 or bias==60: 
            if True:
                avg_values = avg_time_to_0_9[0, 0, bias_idx, plotting_idx] ;std_values = std_time_to_0_9[0, 0, bias_idx, plotting_idx]
                max_val = np.nanmax(avg_values)  ;normalized_avg = avg_values / max_val  ; normalized_std = std_values / max_val
                
                # Plot normalized data in the left subplot (ax[0])
                eb1=ax[0].errorbar(R_list, normalized_avg, yerr=normalized_std, marker='o', linestyle='--', label=f'$\\kappa$={bias}')
                for error_line in eb1[2]:
                    error_line.set_linestyle('--')
                # Plot unnormalized data in the right subplot (ax[1])
                eb2=ax[1].errorbar(R_list, avg_values, yerr=std_values, marker='o',linestyle='--', label=f'$\\kappa$={bias}')
                for error_line in eb2[2]:
                    error_line.set_linestyle('--')


    if False:
        #load data corresponding to vicsek model
        vicsek_time_to_reach_0_9=np.load(path+'/2025-02-18_vicsek'+'/time_to_reach_0_9.npy') ;vicsek_r=np.load(path+'/2025-02-18_vicsek'+'/R_list.npy')
        vicsek_time_to_reach_0_9=np.array(vicsek_time_to_reach_0_9,dtype=float)  ; vicsek_time_to_reach_0_9_avg=np.nanmean(vicsek_time_to_reach_0_9,axis=4)
        vicsek_max_value=np.nanmax(vicsek_time_to_reach_0_9_avg[0,0,1][1:-2]) 
        print(vicsek_r,vicsek_time_to_reach_0_9_avg[0,0,1])
        ax[0].plot(vicsek_r[1:-2], vicsek_time_to_reach_0_9_avg[0,0,1][1:-2]/vicsek_max_value, marker='+', label='Vicsek',linewidth=2)
        

    # Customize both subplots
    ax[0].set_title("Normalized Time-to-Reach 0.9")
    ax[1].set_title("Unnormalized Time-to-Reach 0.9")
    for a in ax:
        a.set_xlabel("R")
        a.tick_params(direction='out', labelsize=10, size=5, width=2, pad=5)
        #a.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        a.legend(frameon=False, fontsize=7)
    
    ax[0].set_ylabel("Normalized Time-steps")
    ax[1].set_ylabel("Time-steps")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "optimal_off_combined.svg"), format="svg", bbox_inches='tight')
    plt.savefig(os.path.join(output_path, "optimal_off_combined.pdf"), format="pdf", bbox_inches='tight')
    plt.show()

