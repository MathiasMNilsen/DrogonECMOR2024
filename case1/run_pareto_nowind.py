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
    np.save('windpower', np.zeros_like(np.load('../data/windpower.npy')))
    kw_sim['num_profiles'] = 1
    kw_sim['path_to_windpower'] = 'windpower.npy'
    
    def dummy_function(*args):

        # get intesitys
        intensity = co2_per_toe(*args, save_emissions=True, save_npv=True)

        # get npv
        npv = np.load('npv.npy')/(-1.0e11)

        # get emissions
        em = np.load('co2_emissions.npy')/1000_000

        # calc obj value
        obj_value = weight*em + (1-weight)*npv
        
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

def run_no_wind(weight, save_folder):

    _delete_En_folders()
    
    # read config file
    kw_opt, kw_sim, kw_ens = read_config.read_toml('popt_config.toml')

    # set save folder
    kw_opt['save_folder'] = save_folder

    # cofig no wind
    np.save('windpower', np.zeros_like(np.load('../data/windpower.npy')))
    kw_sim['num_profiles'] = 1
    kw_sim['path_to_windpower'] = 'windpower.npy'

    def pareto_obj(pred_data, keys_opt, report):
        
        # get intesity
        intensity = co2_per_toe(pred_data, keys_opt, report, save_npv=True, save_emissions=True)

        # get npv
        npv = np.load('npv.npy')/(-1.0e11)
        os.remove('npv.npy')

        # get emissions
        em = np.load('co2_emissions.npy')
        if len(em.shape) == 1:
            em = np.sum(em)/1000_000
        else:
            em = np.sum(em, axis=1)/1000_000
        os.remove('co2_emissions.npy')

        obj_value = weight*em + (1-weight)*npv

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

    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        run_no_wind(weight=w, save_folder=f'pareto_npv_vs_em_nowind/weight_{w}')