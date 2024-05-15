''' This script sets up the windpower ensemble form the NORA10EI data set'''

import numpy  as np
import pandas as pd

def power_curve(x):
    '''
    Power curve of 8MW wind-turbine

    Units: [m/s] --> [MW]

    Source: https://nrel.github.io/turbine-models/2016CACost_NREL_Reference_8MW_180.html
    '''
    from scipy.interpolate import interp1d

    # load data from GitHub
    url = 'https://raw.githubusercontent.com/NREL/turbine-models/master/Offshore/2016CACost_NREL_Reference_8MW_180.csv'
    curve_data = pd.read_csv(url)
    wind_speed = curve_data[curve_data.columns[0]].values # units: [m/s]
    pow_output = curve_data[curve_data.columns[1]].values # units: [kW]

    # interpolate
    curve = interp1d(x=wind_speed,
                     y=pow_output,
                     fill_value=(0.0, 0.0),
                     bounds_error=False)
    
    return curve(x)/1000 # convert: [kW] --> [MW]
    


if __name__ == '__main__':
    
    #internal import
    import windtools as tools
   
    # location of Hywind Tampen
    lat = 61.3338
    lon = 2.2594

    # this is not actually the timeframe of the Drogon field
    start_date = '2010-08-01' 
    end_date   = '2015-01-01'

    # NORA10EI
    ws_df = tools.load_NORA10E_windspeed_data(latitude=lat,
                                              longitude=lon,
                                              height=80,
                                              start_time=start_date,
                                              end_time=end_date,
                                              daily_mean=True)

    # generate windpseed realizations
    ws_en, trace = tools.Generate_Windspeed_Realizations(data=ws_df['windspeed'].values,
                                                         ensemble_size=100)
    
    trace_df = trace.to_dataframe()
    trace_df.to_csv('data/trace')
    
    # save data
    np.save('data/windspeed_ensemble', ws_en)
    
    # make windpower ensemble 
    n_turbines = 1
    pw_en = power_curve(x=np.load('data/windspeed_ensemble.npy'))
    np.save('data/windpower_ensemble', pw_en*n_turbines)