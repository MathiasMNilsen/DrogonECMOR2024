import os
import shutil
import numpy as np

from glob import glob

# import PET
from popt.update_schemes.enopt import EnOpt
from popt.loop.ensemble import Ensemble
from simulator.opm import flow
from input_output import read_config

from objective import objectives

def _delete_En_folders():
    for folder in glob('En_*'):                                             
        shutil.rmtree(folder)

def evaluate(weight, save_folder):

    # read config file
    kw_opt, kw_sim, kw_ens = read_config.read_toml('popt_config.toml')

    # set stuffdf
    kw_opt['save_folder'] = save_folder
    kw_ens['ne'] = 1   
    np.save('windpower', np.zeros_like(np.load('../data/windpower.npy')))
    kw_sim['num_profiles'] = 1
    kw_sim['path_to_windpower'] = 'windpower.npy'

    def dummy_pareto_function(pred_data, keys_opt, report):
        intensity, npv, emissions = objectives(pred_data, keys_opt, report)
        function_value = weight*emissions.sum(axis=1)/1e5 + (1-weight)*npv/(-1.0e10)

        # save stuff
        np.savez(f'{save_folder}/eval', 
                 obj=function_value, 
                 pred_data=pred_data, 
                 emissions=emissions,
                 intensity=intensity,
                 npv=npv)

        return function_value
    
    # get last iteration
    last_iter  = len([f for f in os.listdir(save_folder) if 'optimize_result' in f]) - 1
    state_eval = np.load(f'{save_folder}/optimize_result_{last_iter}.npz')['x']
    
    # evaluate
    Ensemble(kw_ens, flow(kw_sim), dummy_pareto_function).function(x=state_eval)


def run(weight, save_folder):

    _delete_En_folders()
    
    # read config file
    kw_opt, kw_sim, kw_ens = read_config.read_toml('popt_config.toml')

    # set save folder
    kw_opt['save_folder'] = save_folder

    # cofig no wind
    np.save('windpower', np.zeros_like(np.load('../data/windpower.npy')))
    kw_sim['num_profiles'] = 1
    kw_sim['path_to_windpower'] = 'windpower.npy'

    def pareto_function(pred_data, keys_opt, report):
        intensity, npv, emissions = objectives(pred_data, keys_opt, report)
        function_value = weight*emissions.sum(axis=1)/1e5 + (1-weight)*npv/(-1.0e10)
        return function_value

    # optimize    
    ensemble = Ensemble(kw_ens, flow(kw_sim), pareto_function)
    EnOpt(fun=ensemble.function,
          x=ensemble.get_state(),
          args=(ensemble.get_cov(),),
          jac=ensemble.gradient,
          hess=ensemble.hessian,
          bounds=ensemble.get_bounds(),
          **kw_opt)
    


if __name__ == '__main__':
    
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        
        run(weight=w, save_folder=f'pareto_nowind/weight_{w}')
        evaluate(weight=w, save_folder=f'pareto_nowind/weight_{w}')