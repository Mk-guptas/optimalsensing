# plotting the anlaysis from the saved data
import pathlib

path = pathlib.PureWindowsPath(r"E:\Manish\simulation_result\optimal_sensing").as_posix()

#output_path=path+'/2025-04-07_vicsek_model'
#output_path=path+'/2025-04-08_pure_votermodel'
#output_path=path+'/2025-04-08_LEUP_uniform_onlattice'
#output_path=path+'/2025-04-25_LEUP_memory_onlattice'
output_path=path+'/2025-05-20_LEUP_von_misses_offlattice_biased_entropy'


#plotting polar order
#polar_order_plotter(output_path)


time_to_reach_consensus(output_path,bias_idx=0)
#plotting phase plot
normalized_time_consensus(output_path,path,'bias',plotting_idx=0)
