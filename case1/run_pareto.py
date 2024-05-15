import os
import shutil
import numpy as np

from glob import glob

# import PET
from popt.update_schemes.linesearch import LineSearch
from popt.loop.ensemble import Ensemble
from simulator.opm import flow
from input_output import read_config

from objective import co2_per_toe

def _delete_En_folders():
    for folder in glob('En_*'):                                             
        shutil.rmtree(folder)

def evaluate(weight, save_folder):

    # read config file
    kw_opt, kw_sim, kw_ens = read_config.read_toml('popt_config.toml')

    # set stuffdf
    kw_opt['save_folder'] = save_folder
    kw_ens['ne'] = 1   
    kw_sim['path_to_windpower'] = '../data/windpower_ensemble_eval.npy'
    
    def dummy_function(*args):

        # get intesitys
        intensity = co2_per_toe(*args, save_emissions=True, save_npv=True)

        # get npv
        npv = np.load('npv.npy')/(-1.0e9)


        obj_value = weight*intensity + (1-weight)*npv
        
        # save
        np.savez(f'{save_folder}/final_run', 
                 obj=obj_value, 
                 pred_data=args[0], 
                 emissions=np.load('co2_emissions.npy'),
                 intensity=intensity,
                 npv=np.load('npv.npy'))
        
        # delete co2 array
        os.remove('co2_emissions.npy')
        os.remove('npv.npy')

        return obj_value
    
    # get last iteration
    last_iter = len([f for f in os.listdir(save_folder) if 'optimize_result' in f]) - 1
    x_eval = np.load(f'{save_folder}/optimize_result_{last_iter}.npz')['x']
    
    # evaluate
    Ensemble(kw_ens, flow(kw_sim), dummy_function).function(x=x_eval)


def run(weight, save_folder):

    _delete_En_folders()
    
    # read config file
    kw_opt, kw_sim, kw_ens = read_config.read_toml('popt_config.toml')

    # set save folder
    kw_opt['save_folder'] = save_folder

    def pareto_obj(pred_data, keys_opt, report):
        
        # get intesity
        intensity = co2_per_toe(pred_data, keys_opt, report, save_npv=True)

        # get npv
        npv = np.load('npv.npy')/(-1.0e9)
        os.remove('npv.npy')

        obj_value = weight*intensity + (1-weight)*npv

        return obj_value

    # optimize    
    ensemble = Ensemble(kw_ens, flow(kw_sim), pareto_obj)
    LineSearch(fun=ensemble.function,
               x=ensemble.get_state(),
               args=(ensemble.get_cov(),),
               jac=ensemble.gradient,
               hess=ensemble.hessian,
               bounds=ensemble.get_bounds(),
               **kw_opt)
    
    evaluate(weight, save_folder)


if __name__ == '__main__':

    # set initial rates
    np.savez('init_injrate', 6000*np.ones_like(np.load('init_injrate.npz')['arr_0']))
    np.savez('init_prodrate', 13_000*np.ones_like(np.load('init_prodrate.npz')['arr_0']))

    for w in [0.7, 0.8, 0.9, 1.0]:
        run(weight=w, save_folder=f'pareto2/weight_{w}')
