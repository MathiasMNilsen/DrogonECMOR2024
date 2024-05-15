import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mplcyberpunk
import os

plt.style.use("cyberpunk")
mpl.rcParams.update({'text.color' : 'black',
                     'axes.labelcolor' : 'black',
                     'xtick.color' : 'black',
                     'ytick.color' : 'black',
                     'grid.linestyle' : '-',
                     'grid.color' : 'gray',
                     'grid.alpha' : 0.4,
                     'figure.facecolor' : 'white',
                     'axes.facecolor' : 'white',
                     'savefig.facecolor': 'white'})

COLOR_MAP = 'Spectral'
DATES = pd.date_range('2020-08-01', '2025-01-01', freq='MS')

def run_calc(prod_data, windpower=None):

    if windpower is None:
         windpower = np.zeros(len(prod_data)-1)

    from libecalc.application.energy_calculator import EnergyCalculator
    from libecalc.common.time_utils import Frequency
    from libecalc.presentation.yaml.model import YamlModel

    from pathlib import Path
    HERE = Path().cwd()  # fallback for ipynb's

    data_getter = lambda i, key: prod_data[i+1][key].squeeze() - prod_data[i][key].squeeze()
    eCalc_input = {'dd-mm-yyyy' : [],
                  'OIL_PROD'   : [],
                    'GAS_PROD'   : [], 
                    'WATER_INJ'  : [],
                    'THP_MAX'    : [],
                    'WIND_POWER' : []}
    
    for t in range(len(prod_data)-1):
            
            eCalc_input['dd-mm-yyyy'].append(DATES[t+1].strftime('%m-%d-%Y'))
            eCalc_input['OIL_PROD'].append(data_getter(t, 'fopt')/(DATES[t].days_in_month))
            eCalc_input['GAS_PROD'].append(data_getter(t, 'fgpt')/(DATES[t].days_in_month))
            eCalc_input['WATER_INJ'].append(data_getter(t, 'fwit')/(DATES[t].days_in_month))
            eCalc_input['WIND_POWER'].append(windpower[t]) 

            # get maximum well head pressure (for each ensemble member)
            thp_vals = []
            for key in ['wthp a5', 'wthp a6']:
                thp_vals.append(prod_data[t][key].squeeze())

            eCalc_input['THP_MAX'].append(np.max(np.array(thp_vals), axis=0))


    # run eCalc
    pd.DataFrame(eCalc_input).to_csv('ecalc_input.csv', index=False)
    yaml_model = YamlModel(path=HERE/'ecalc_config.yaml', output_frequency=Frequency.NONE)
    model = EnergyCalculator(graph=yaml_model.graph)

    consumer_results = model.evaluate_energy_usage(yaml_model.variables)
    emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)
    os.remove('ecalc_input.csv')

    # extract results
    ecalc_res = {'power_demand' : 0.0,
                 'fuel_rate'    : 0.0}

    
    for identity, component in yaml_model.graph.nodes.items():

        if identity in consumer_results:
            res = consumer_results[identity].component_result
            if res.energy_usage.unit == 'MW':
                ecalc_res['power_demand'] += np.asarray(res.energy_usage.values)
            if res.energy_usage.unit == 'Sm3/d':
                ecalc_res['fuel_rate'] = np.array(res.energy_usage.values)
    
    fuel_volume_monthly = pd.Series(ecalc_res['fuel_rate'], 
                                    index=DATES[:-1]).resample('MS').sum()

    ecalc_res['fuel_vol'] = fuel_volume_monthly

    return ecalc_res

def analyze_omega_1():
    co2_tax = 0.15 # USD/kg
    co2_em_factor = 2.416  #kg CO2/Sm3
    gas_price = 17 # USD/Sm3
    
    file_nw = np.load('pareto_nowind/weight_0.0/eval.npz', allow_pickle=True)
    file_w  = np.load('pareto_wind/weight_0.0/eval_50.npz', allow_pickle=True)

    delta_NPV = np.squeeze(file_w['npv'].mean() - file_nw['npv'])/1e9
    print('NPV wind: ', file_w['npv'].mean()/1e9)
    print('NPV gas: ', file_nw['npv'].squeeze()/1e9)
    print('Δ: ', delta_NPV, ' Billion USD')
    print('\n')

    delta_EM = np.squeeze(file_nw['emissions'].sum()-file_w['emissions'].sum(1).mean())/1000
    print('CO2 wind: ', file_w['emissions'].sum(1).mean()/1000)
    print('CO2 gas: ', file_nw['emissions'].sum()/1000)
    print('ΔCO2: ', delta_EM, ' kilo tonnes') # kilo ton
    print('ΔUSD: ', co2_tax*delta_EM, ' Million USD')
    print(r'ΔV_fuel: ', delta_EM/co2_em_factor * gas_price, ' Million USD')

def analyze_wind(w):

    file = np.load(f'pareto_wind/weight_{w}/eval_50.npz', allow_pickle=True)

    npv = file['npv'].mean()/1e9                # billion USD
    co2 = file['emissions'].sum(1).mean()/1000  # kilo tonnes
    iem = file['intensity'].mean()              # kg per toe

    print('NPV: ', np.round(npv, 1), 'billion USD')
    print('CO2: ', np.round(co2, 1), 'kilo tonnes')
    print('Intensity: ', np.round(iem, 1), 'kg/toe')

#analyze_omega_1()
analyze_wind(0.7)
