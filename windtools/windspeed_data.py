import xarray as xr
import numpy  as np
import pandas as pd

from datetime import timedelta
from pyproj   import Transformer
from tqdm     import tqdm

__all__ = ['load_NORA10E_windspeed_data'] 

def trafo2rotated(lat, lon):
    trafo_str = '+proj=ob_tran +o_proj=longlat +lon_0=-40 +o_lat_p=22 +a=6367470 +e=0'
    trans = Transformer.from_crs('epsg:4326', trafo_str, always_xy=True)
    lon, lat = trans.transform(lon, lat)
    return lat, lon


def load_NORA10E_windspeed_data(latitude, longitude, height=80, start_time='2017-01-01', end_time='2017-12-31', daily_mean = False):
    '''
    Load wind speed data from the NORA10E dataset for a specified location and time range.
    WARNING: This takes some time!!

    Parameters
    ----------
        latitude : float
            Latitude of the location of interest.

        longitude : float
            Longitude of the location of interest.

        height : int (optional)
            Height in meters above ground for wind speed data (default is 80m).
            Possible values are 30, 50, 80, 150, 200

        start_time : str (optional)
            Start date for the time range in the format 'YYYY-MM-DD' 
            (default is '2017-01-01'). The oldest date is 1979-01-01. 

        end_time : str (optional)
            End date for the time range in the format 'YYYY-MM-DD' 
            (default is '2017-12-31'). The most rescent date is 2017-12-31.
        
        daily_mean : bool (optional)
            If True, only the mean windspeed for each day is given. If False
            windspeed for each 3rd hours is given (default is False)

    Returns
    -------
        pandas.DataFrame: A DataFrame containing time and wind speed data for the specified location and time range.

    Example
    -------
        To load wind speed data for a location at latitude 54.0, longitude 3.0, at 80 meters above ground,
        for the time range from '2023-01-01' to '2023-01-31', you can call the function like this:
        >>> data = load_NORA10E_windspeed_data(54.0, 3.0, height=80, start_time='2023-01-01', end_time='2023-01-31')
    '''
    if height in [30, 50, 80]:
        height = f'0{height}' # put a zero before

    # Load the dataset from the URL
    url  = f'https://thredds.met.no/thredds/dodsC/nora10ei_wind/NORA10EI_windspeed_{height}m.ncml'
    data = xr.open_dataset(url)
    data = data[f'windspeed_{height}m']
    
    # Check if dates make sense and are in range
    ###############################################################################################################
    old = np.datetime64('1979-01-01')
    new = np.datetime64('2017-12-31')

    start = np.datetime64(start_time)
    end   = np.datetime64(end_time)

    start_in_range = old <= start <= new
    end_in_range   = old <= end   <= new

    if start_in_range and end_in_range and start < end:
        # Construct time interval
        time = np.arange(start, end+np.timedelta64(1, 'D'), timedelta(hours=3)) # 3 hours is the time resolution of the NORA10E dataset 
    else:
        # Raise errors
        if not start_in_range:
            raise ValueError('start_time is not in range 1979-01-01 to 2017-12-31')
        elif not end_in_range:
            raise ValueError('end_time is not in range 1979-01-01 to 2017-12-31')
        else:
            raise ValueError('end_time before start_time')
    ###############################################################################################################

    # Select data
    rlat, rlon = trafo2rotated(latitude, longitude)
    wind_speed_raw = data.sel(rlat=rlat,
                              rlon=rlon,
                              time=time,
                              method='nearest')

    # Make pandas DataFrame
    ###############################################################################################################
    if daily_mean:

        # Calculate mean of each days
        days = np.arange(start, end+np.timedelta64(1, 'D'), timedelta(days=1))
        pbar = tqdm(range(days.size), desc='Preparing data: ', ncols=100) # progess bar
        wind_speed_values = np.zeros(days.size)

        for d in pbar:
            index = d*8
            wind_speed_values[d] = np.mean(wind_speed_raw[index:index+8].values)
        time = days
        
    else:
        wind_speed_values = np.zeros(time.size)
        pbar = tqdm(time, desc='Preparing data: ', ncols=100)
        for i, _ in enumerate(pbar):

            wind_speed_values[i] = wind_speed_raw[i].values

    wind_speed_dataframe = pd.DataFrame({'time': time, 'windspeed': wind_speed_values})
    wind_speed_dataframe = wind_speed_dataframe.set_index('time')
    ###############################################################################################################

    return wind_speed_dataframe