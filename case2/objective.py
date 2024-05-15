import pandas as pd
import numpy  as np
import os
import yaml
import time

from pathlib import Path
from multiprocessing import Pool

HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()

def objectives(pred_data, keys_opt, report):
    '''
    Parameters
    ----------
    pred_data : array_like
        Ensemble of predicted data.

    keys_opt : list
        Keys with economic data.

    report : list
        Report dates.

    Returns
    -------
    objective_values : array_like
        Objective function values (NPV) for all ensemble members.
    '''
    global const
    global report_dates
    global sim_kwargs
    global sim_data
  
    # define a data getter
    get_data = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()

    # ensemble size (ne), number of report-dates (nt)
    nt = len(pred_data) 
    try: ne = len(get_data(1,'fopt'))
    except: ne = 1

    # Economic and other constatns
    const = dict(keys_opt['npv_const'])
    sim_kwargs = keys_opt
    report_dates = report[1]

    sim_data = {'fopt': np.zeros((ne, nt-1)),
                'fgpt': np.zeros((ne, nt-1)),
                'fwpt': np.zeros((ne, nt-1)),
                'fwit': np.zeros((ne, nt-1)),
                'thp' : np.zeros((ne, nt-1)),
                'days': np.zeros(nt-1)}
    
    # loop over pred_data
    for t in range(nt-1):
        
        for datatype in ['fopt', 'fgpt', 'fwpt', 'fwit']:
            sim_data[datatype][:,t] = get_data(t, datatype)
        
        # days in time-step
        sim_data['days'][t] = (report_dates[t+1] - report_dates[t]).days

        # get maximum well head pressure (for each ensemble member)
        thp_keys = [k for k in keys_opt['datatype'] if 'wthp' in k] # assume only injection wells
        thp_vals = []
        for key in thp_keys:
            thp_vals.append(pred_data[t][key].squeeze())
            
        sim_data['thp'][:,t] = np.max(np.array(thp_vals), axis=0)

    # calculate NPV values
    if ne == 1 and sim_kwargs['num_profiles'] > 1:
        indecies = [[n,0] for n in range(sim_kwargs['num_profiles'])]
    elif ne > 1 and sim_kwargs['num_profiles'] == 1:
        indecies = [[0,n] for n in range(ne)] 
    else:
        indecies = [[n,n] for n in range(sim_kwargs['num_profiles'])]

    with Pool(keys_opt['parallel']) as pool:
        values = pool.map(_objective, indecies)

    intensity = [val[0] for val in values]
    npv = [val[1] for val in values]
    emissions  = [val[2] for val in values]
    
    return np.asarray(intensity), np.asarray(npv), np.asarray(emissions)



def _objective(args):
    nw, nc = args

    # load windpower
    try:
        wind_power = np.load(sim_kwargs['path_to_windpower'])[nw]
    except:
        time.sleep(2)
        wind_power = np.load(sim_kwargs['path_to_windpower'])[nw]
        

    eCalc_data = get_eCalc_data(nc, wind_power)

    # calc co2 emissions with eCalc
    emissions_rate_daily, fuel_rate_daily = run_eCalc(max(nw, nc), eCalc_data) # ton/day

    # resample emissions to monthly values
    index = pd.date_range(report_dates[0], report_dates[-1], freq='D')
    emissions_volume_monthly = pd.Series(emissions_rate_daily, index=index[:-1]).resample('MS').sum()
    fuel_volume_monthly = pd.Series(fuel_rate_daily, index=index[:-1]).resample('MS').sum()

    oil_export = sim_data['fopt'][nc]
    gas_export = sim_data['fgpt'][nc] - fuel_volume_monthly.values
    water_prod = sim_data['fwpt'][nc]
    water_inj  = sim_data['fwit'][nc]

    # calc kg co2 per toe
    co2  = emissions_volume_monthly*1000    # ton --> kg
    fopt = oil_export*0.84                  # Sm³ --> toe (ton of oil equivalent)
    fgpt = sim_data['fgpt'][nc]*0.84/1000   # Sm³ --> toe (ton of oil equivalent)
    
    co2_per_toe_value = np.sum(co2)/(fopt.sum()+fgpt.sum())

    
    # calc NPV
    revenue   = const['wop']*oil_export + const['wgp']*np.clip(gas_export, 0, None)
    expenses  = const['wwp']*water_prod + const['wwi']*water_inj + const['wem']*emissions_volume_monthly
    discount  = (1+const['disc'])**(sim_data['days']/365)
    npv_value = np.sum( (revenue-expenses)/discount )  

    return co2_per_toe_value, npv_value, emissions_volume_monthly


def run_eCalc(n: int, ecalc_data: dict):
    from libecalc.application.energy_calculator import EnergyCalculator
    from libecalc.common.time_utils import Frequency
    from libecalc.presentation.yaml.model import YamlModel
    
    pd.DataFrame(ecalc_data).to_csv(f'ecalc_input_{n}.csv', index=False)
    new_yaml = duplicate_yaml_file(sim_kwargs['ecalc_yamlfile'], n)

    # Config
    yaml_model = YamlModel(path=HERE/new_yaml, output_frequency=Frequency.NONE)

    # Compute energy, emissions
    model = EnergyCalculator(graph=yaml_model.graph)
    consumer_results = model.evaluate_energy_usage(yaml_model.variables)
    emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

    # get fuel
    fuel = []
    for id in consumer_results:
        if type(yaml_model.graph.get_node(id)).__name__ == 'GeneratorSet':
            fuel.append(consumer_results[id].component_result.energy_usage.values)
    
    # get emissions
    co2 = []
    for id_hash in emission_results:
        res = emission_results[id_hash]
        co2.append(res['co2_fuel_gas'].rate.values)

    fuel = np.sum(np.asarray(fuel), axis=0) # Sm³/day
    co2 = np.sum(np.asarray(co2), axis=0)   # tons/day

    # delete dummy files
    os.remove(new_yaml)
    os.remove(f'ecalc_input_{n}.csv')

    return co2, fuel


def duplicate_yaml_file(filename, member):

    # Load the YAML file
    try:
        with open(filename, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
    except:
        time.sleep(2)

    input_name = data['TIME_SERIES'][0]['FILE']
    data['TIME_SERIES'][0]['FILE'] = input_name.replace('.csv', f'_{member}.csv')

    # Write the updated content to a new file
    new_filename = filename.replace(".yaml", f"_{member}.yaml")
    with open(new_filename, 'w') as new_yaml_file:
        yaml.dump(data, new_yaml_file, default_flow_style=False)

    return new_filename


def get_eCalc_data(n, power_wind):

    # dates
    daily_dates = pd.date_range(report_dates[0], report_dates[-1], freq='D')

    data = {'dd-mm-yyyy' : [],
            'OIL_PROD'   : [],
            'GAS_PROD'   : [], 
            'WATER_INJ'  : [],
            'THP_MAX'    : [],
            'WIND_POWER' : []}

    # index
    month = daily_dates[0].month
    month_index = 0

    # loop over days
    for day_index, day in enumerate(daily_dates[:-1]):
        
        if not month == day.month:
            month = day.month
            month_index += 1

        data['dd-mm-yyyy'].append(day.strftime('%d-%m-%Y'))
        data['OIL_PROD'].append(sim_data['fopt'][n, month_index]/sim_data['days'][month_index])
        data['GAS_PROD'].append(sim_data['fgpt'][n, month_index]/sim_data['days'][month_index])
        data['WATER_INJ'].append(sim_data['fwit'][n, month_index]/sim_data['days'][month_index])
        data['THP_MAX'].append(sim_data['thp'][n, month_index])
        data['WIND_POWER'].append(-power_wind[day_index])
   
    return data

