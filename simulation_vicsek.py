# Start timer
start_time = time.time()
# Parameters
model_type='vicsek_model'

# Define output path
path = pathlib.PureWindowsPath(r"E:\Manish\simulation_result\optimal_sensing").as_posix()
date = str(datetime.now().date())
output_path = os.path.join(path, date+'_'+str(model_type))
os.makedirs(output_path, exist_ok=True)

bias_param_list=[0]
beta_list = [0,1,50] 
#beta_list = [0,1,5,10,50]
R_list = [0.5,1,2,3,4]

model_parameters={'beta_list':beta_list,'R_list':R_list,'bias_param_list':bias_param_list,'speed_list':[0.1],'L_list':[10],\
                    'simulation_steps':1000,'eta':0,'No_of_particle':250,'no_of_trajectory':2}

simulator_offlattice(model_type,model_parameters,output_path)

# Print execution time
end_time = time.time()
print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")
