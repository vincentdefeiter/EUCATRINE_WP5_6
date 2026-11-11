'''
EU-CATRINE Testbed Functions

Author: Vincent de Feiter (vincent.defeiter@wur.nl)

Version: 11 November, 2026

'''

#Import Modules

#Processing
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy import stats
import json
import os

#Visualisation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms as transforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from matplotlib import patheffects
from matplotlib import rcParams
import matplotlib.patheffects as path_effects

import sys
sys.path.append('/home/vincent/EU-CATRINE-WP5_6/Testbed - Amazon/code-utils')

#-----------------------------------------------
#                G E N E R A L 
#-----------------------------------------------

#Round a value to the nearest base, but only if the value is close enough to the rounded value.
def round_if_close(val, base=5, threshold=3):
    
    '''
    Round a value to the nearest base, but only if the value is close enough to the rounded value.
    Parameters:
    val : float
        The value to be rounded.
    base : float, optional
        The base to which the value should be rounded (default is 5).
    threshold : float, optional
        The threshold within which the value is considered close enough to the rounded value (default is 3).
    Returns:
    float
        The rounded value if it is close enough to the rounded value, otherwise the original value.
    '''
    
    rounded = round(val / base) * base
    return rounded if abs(val - rounded) <= threshold else round(val)


#Find the closest index in an array to a given value
def find_closest(arr, val):
    '''
    Find the index of the closest value in an array to a given value.
    Parameters:
    arr : array-like
        The array in which to search for the closest value.
    val : float
        The value to which the closest value in the array should be found.
    Returns:
    int
        The index of the closest value in the array to the given value.
    '''
    
    idx = np.abs(arr - val).argmin()
    return idx

#Align two y-axes so that a specific value on one axis visually aligns with a specific value on the other axis.
def align_yaxis(ax1, val1, ax2, val2):
    """
    Align val1 on ax1 to match val2 on ax2 (visually at the same height).

    Parameters:
    - ax1: First axis (e.g., main axis)
    - val1: Y-value on ax1 to align
    - ax2: Second axis (e.g., twinx axis)
    - val2: Y-value on ax2 to align with val1
    """
    # Get current limits
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    # Calculate the position of val1 and val2 in figure coordinates
    inv1 = ax1.transData.transform
    inv2 = ax2.transData.inverted().transform

    # Find display coordinate of val1
    coord = inv1((0, val1))

    # Convert that display coordinate into data coordinate on ax2
    aligned_val2 = inv2(coord)[1]

    # Adjust ax2 limits so val2 appears at the same position as val1
    dy = val2 - aligned_val2
    ax2.set_ylim(y2_min + dy, y2_max + dy)

#Make a legend with a specified number of rows and columns, and position it at a specific location
def legend_maker(axis,nrows,ncols,x,y):
    
    # Extract the handles and labels
    handles, labels = axis.get_legend_handles_labels()

    # Specify the number of rows and columns
    n_rows = nrows
    n_cols = ncols 

    # Rearrange handles and labels
    sorted_handles_labels = []
    for col in range(n_cols):
        for row in range(n_rows):
            index = row * n_cols + col
            if index < len(handles):
                sorted_handles_labels.append((handles[index], labels[index]))

    # Unpack sorted handles and labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    axis.legend(sorted_handles,sorted_labels,loc='lower left', bbox_to_anchor= (x, y), ncols=n_cols,
                    borderaxespad=0, frameon=False) 


#-----------------------------------------------
#            S T A T I S T I C S 
#-----------------------------------------------

# Calculate mean, ignoring NaN values
def calc_var(x):
    """
    Calculate variance, ignoring NaN values
    """
    return 1/np.sum(~np.isnan(x)) * np.nansum((x - np.nanmean(x))**2)

# Calculate standard deviation, ignoring NaN values
def calc_stddev(x):
    """
    Calculate standard deviation, ignoring NaN values
    """
    return np.sqrt(calc_var(x))

# Calculate Pearson correlation coefficient, ignoring NaN values
def calc_r(r, f):
    """
    Calculate correlation coefficient, ignoring NaN values
    """
    # Mask NaN values in both arrays
    mask = ~np.isnan(r) & ~np.isnan(f)
    r, f = r[mask], f[mask]
    
    var_r = calc_stddev(r)
    var_f = calc_stddev(f)
    return 1./r.size * np.nansum((f - np.nanmean(f)) * (r - np.nanmean(r))) / (var_f * var_r)

# Calculate Pearson correlation coefficient, ignoring NaN values, and return angle in radians
def calc_rmse(r, f):
    """
    Calculate RMS error (including mean error), ignoring NaN values
    """
    # Mask NaN values in both arrays
    mask = ~np.isnan(r) & ~np.isnan(f)
    r, f = r[mask], f[mask]
    
    return np.sqrt(1./r.size * np.nansum((f - r)**2))

# Calculate centered RMS error (excluding mean error), ignoring NaN values
def calc_crmse(r, f):
    """
    Calculate centered RMS error (excluding mean error), ignoring NaN values
    """
    # Mask NaN values in both arrays
    mask = ~np.isnan(r) & ~np.isnan(f)
    r, f = r[mask], f[mask]
    
    return np.sqrt(1./r.size * np.nansum(((f - np.nanmean(f)) - (r - np.nanmean(r)))**2))

# Calculate mean bias, ignoring NaN values
def calc_bias(r, f):
    """
    Calculate mean bias, ignoring NaN values
    """
    # Mask NaN values in both arrays
    mask = ~np.isnan(r) & ~np.isnan(f)
    r, f = r[mask], f[mask]
    
    return np.nanmean(f) - np.nanmean(r)

# Calculate the angle (in radians) from the correlation coefficient
def r_to_theta(r):
    """
    Relation r-coefficient and theta (angle, radians)
    """
    return np.arccos(r)

# Convert polar coordinates (radius and angle) to Cartesian coordinates (x, y)
def to_xy(r, theta):
    """
    radius and angle (radians) to x,y coordinates
    """
    return r*np.cos(theta), r*np.sin(theta)

# Calculate peak-to-peak amplitude of a signal, ignoring NaN values
def calc_amplitude(x):
    """
    Calculate peak-to-peak amplitude of a signal, ignoring NaN values.
    """
    return np.nanmax(x) - np.nanmin(x)

# Align two datasets based on the x values of the reference dataset
def align_data_to_reference(x_ref, x_data, y_ref, y_data):
    """
    Aligns y_data to x_ref axis (cuts y_data to x_ref points, matching nearest x_data).
    
    Parameters:
    x_ref (array-like): Reference x-axis (e.g., DALES heights).
    x_data (array-like): x-axis of the data to be aligned (e.g., OBS heights).
    y_ref (array-like): y values of the reference data (e.g., DALES data).
    y_data (array-like): y values of the data to be aligned (e.g., OBS data).
    
    Returns:
    tuple: x_ref, y_ref, y_data_aligned
    """
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_data = np.asarray(x_data, dtype=np.float64)
    y_ref = np.asarray(y_ref)
    y_data = np.asarray(y_data)

    df_ref = pd.DataFrame({'x': x_ref, 'y_ref': y_ref})
    df_data = pd.DataFrame({'x': x_data, 'y_data': y_data})

    df_merged = pd.merge_asof(df_ref.sort_values('x'),
                              df_data.sort_values('x'),
                              on='x',
                              direction='nearest')

    return df_merged['x'].values, df_merged['y_ref'].values, df_merged['y_data'].values


#-----------------------------------------------
#                   O B S 
#-----------------------------------------------

#Convert a list of time objects to hour fractions (to make reference arrays)
def time_to_hour_fraction(times):  
    '''
    Convert a list of time objects to hour fractions.
    
    Parameters:
    times : list of datetime.time
        List of time objects to be converted.
    
    Returns:
    List of floats representing the hour fraction for each time object.
    '''
    
    return [t.hour + t.minute / 60 for t in times]


#Daily ranges
def daily_range(dates, freq, end="18:00"):
    '''
    Generate a daily time range for a list of dates with a specified frequency.
    
    Parameters:
    dates : list of datetime.date
        List of date objects for which to generate the time range.
    freq : str  
        Frequency string for the time range (e.g., '1min', '5min', etc.).
    end : str, optional
        End time for the daily range (default is "18:00").
    
    Returns:
    pd.DatetimeIndex
        A DatetimeIndex containing the combined date and time for each date in the list.
    '''  
    
    times = pd.date_range("06:00", end, freq=freq).time
    return pd.DatetimeIndex([pd.Timestamp.combine(d, t) for d in dates for t in times])


#Add hour and minute to an xarray Dataset based on a list of dates
def add_hour_minute_to_ds(ds, date_list):
    '''
    Filters an xarray Dataset by given date list and adds a new coordinate `hour_minute`.

    Parameters:
        ds (xarray.Dataset): Dataset with a 'time' coordinate.
        date_list (list of str): List of dates in 'YYYY-MM-DD' format to filter the dataset.

    Returns:
        xarray.Dataset: Filtered dataset with an added 'hour_minute' coordinate (format HHMM).
    '''
    ds = ds.sel(time=ds['time'].dt.strftime('%Y-%m-%d').isin(date_list))
    time_df = ds['time'].to_dataframe()
    time_df['hour_minute'] = time_df.index.hour * 100 + time_df.index.minute
    ds = ds.assign_coords(hour_minute=('time', time_df['hour_minute']))
    return ds


#---------------------------- C L O U D R O O T S - A M A Z O N 2 2   S P E C I F I C ----------------------------

#TOWER DATA

#Process each condition to check for limits and replace with NaN if outside the limits
def process_item(items, upper_limit, lower_limit):
    '''
    Process a list of items, replacing values outside specified limits with NaN.
    
    Parameters:
    items : list or np.array
        List of items to be processed.
    upper_limit : float
        Upper limit for valid values.
    lower_limit : float
        Lower limit for valid values.
    
    Returns:
    list
        A list where items outside the specified limits are replaced with NaN.
    '''
    
    return [item if lower_limit < item < upper_limit else float('nan') for item in items]


# Load tower data from CSV files for specified measurement heights
def load_tower_data(all_heights, tower_data_rootdir):
    """
    Loads tower data CSV files for specified measurement heights.

    Parameters:
    -----------
    all_heights : list of float or int
        List of measurement heights in meters (e.g., [2, 10, 50, 100]).
    tower_data_rootdir : str
        Path to the directory containing the CSV files.
        Expected filenames format: 'DF_XXXm.csv' (e.g., 'DF_010m.csv').

    Returns:
    --------
    df_list : list of pandas.DataFrame
        List of loaded dataframes, one for each height.
    df_name_list : list of str
        List of dataframe names as strings (e.g., ['DF_010m', 'DF_100m']).
    """
    df_list = []
    df_name_list = []

    for height in all_heights:
        stringer = f"{int(height):03d}"  # Zero-padded height (e.g., '002', '010', '100')
        df = pd.read_csv(f"{tower_data_rootdir}/DF_{stringer}m.csv")
        df_list.append(df)
        df_name_list.append(f"DF_{stringer}m")
    
    return df_list, df_name_list

# Function to process each date from tower data
def process_date(date, df_list):
    
    '''
    Process data for a specific date from a list of DataFrames.
    
    Parameters:
    date : str
        The date for which to process the data, in the format 'YYYY-MM-DD'.
    df_list : list of pd.DataFrame
        List of DataFrames containing the data for different heights.
    
    Returns:
    day_data : dict
        A dictionary containing processed data for the specified date, with keys for various meteorological variables.
    
    The values are lists of processed data for each variable, corresponding to the DataFrames in df_list.
    '''
    
    day_data = {
        'time': [],
        'T': [], 'Td': [], 'q': [], 'p': [], 'theta': [], 'TKE': [],
        'H': [], 'LE': [], 'wind_speed': [], 'max_wind_speed': [],
        'wind_u': [], 'wind_v': [], 'wind_dir': [], 'bowen_ratio': [],
        'u_star': [], 'u_var': [], 'v_var': [], 'w_var': [], 'density': [],
        'cp': [], 'VPD': [], 'L': [], 'z_d_L': [], 'wind_w': [], 'NEE': []
    }
    
    for df in df_list:
        df_sub = df.loc[df['date'] == date]
        if df_sub.empty:
            continue
        
        day_data['time'].append(df_sub['time'].values)

        day_data['T'].append(process_item(df_sub['Thermo_T'].astype(float).values, 999, -999))
        day_data['Td'].append(process_item(df_sub['Tdew'].astype(float).values - 273.15, 999, -999))
        day_data['q'].append(process_item(df_sub['specific_humidity'].astype(float).values * 1000, 999, -999))
        day_data['p'].append(df_sub['interpolated_pressure'].astype(float).values/100)
        
        # Potential Temperature Calculation
        T = df_sub['Thermo_T'].astype(float).values
        p = df_sub['interpolated_pressure'].astype(float).values
        cp = df_sub['air_heat_capacity'].astype(float).values
        theta = [(t + 273.15) * ((1000e2 / p[i]) ** (287 / cp[i])) for i, t in enumerate(T)]
        day_data['theta'].append(process_item(theta, 999, -999))

        day_data['TKE'].append(process_item(df_sub['TKE'].astype(float).values, 20, -20))
        day_data['H'].append(process_item(df_sub['H'].astype(float).values, 999, -999))
        day_data['LE'].append(process_item(df_sub['LE'].astype(float).values, 999, -999))                
        day_data['wind_speed'].append(process_item(df_sub['wind_speed'].astype(float).values, 999, -999))
        day_data['max_wind_speed'].append(process_item(df_sub['max_wind_speed'].astype(float).values, 999, -999))
        day_data['wind_u'].append(process_item(df_sub['wind_u_correct'].astype(float).values, 999, -999))
        day_data['wind_v'].append(process_item(df_sub['wind_v_correct'].astype(float).values, 999, -999))
        day_data['wind_dir'].append(process_item(df_sub['wind_dir'].astype(float).values, 999, -999))
        day_data['bowen_ratio'].append(process_item(df_sub['bowen_ratio'].astype(float).values, 20, -20))
        day_data['u_star'].append(process_item(df_sub['u*'].astype(float).values, 1.2, -999))
        day_data['u_var'].append(process_item(df_sub['u_var'].astype(float).values, 999, -999))
        day_data['v_var'].append(process_item(df_sub['v_var'].astype(float).values, 999, -999))
        day_data['w_var'].append(process_item(df_sub['w_var'].astype(float).values, 999, -999))
        day_data['density'].append(process_item(df_sub['air_density'].astype(float).values, 9999, -9999))
        day_data['cp'].append(process_item(df_sub['air_heat_capacity'].astype(float).values, 9999, -9999))
        day_data['VPD'].append(process_item(df_sub['VPD'].astype(float).values / 100, 9999, -99))
        day_data['L'].append(process_item(df_sub['L'].astype(float).values / 1000, 99, -99))
        day_data['z_d_L'].append(process_item(df_sub['(z-d)/L'].astype(float).values, 9999, -9999))
        day_data['wind_w'].append(process_item(df_sub['w_unrot'].astype(float).values, 9999, -9999))
        day_data['NEE'].append(process_item(df_sub['co2_flux'].astype(float).values, 9999, -9999))

    return day_data

#Aggregate tower data across dates and heights
def aggregate_tower_data(data_complete, all_heights, date_list_general, SCu_list, DeepCu_list, slice_range=(20, 45)):
    """
    Aggregates tower measurement data across dates and heights.

    Computes the mean and standard deviation of each variable for:
    - Shallow Cu (SCu) dates
    - Deep Cu dates
    - Daily combined data (all dates)

    Parameters:
    -----------
    data_complete : dict
        Dictionary containing processed data with keys for each variable.
        Structure: data_complete['variable'][date_idx][height_idx] = time-series data.
    all_heights : list of float or int
        List of measurement heights corresponding to the data.
    date_list_general : list of str or datetime
        List of all dates processed (order must match data_complete).
    SCu_list : list of str or datetime
        List of dates classified as Shallow Cu (SCu) cases.
    DeepCu_list : list of str or datetime
        List of dates classified as Deep Cu cases.
    slice_range : tuple of int, optional (default=(20, 45))
        Tuple indicating the range of indices to slice from each time-series before aggregation.

    Returns:
    --------
    SCu_tower_mean : dict
        Dictionary with SCu mean arrays for each variable [height, time-series].
    SCu_tower_std : dict
        Dictionary with SCu std deviation arrays for each variable [height, time-series].
    DeepCu_tower_mean : dict
        Dictionary with Deep Cu mean arrays for each variable [height, time-series].
    DeepCu_tower_std : dict
        Dictionary with Deep Cu std deviation arrays for each variable [height, time-series].
    daily_tower : dict
        Dictionary with combined daily arrays for each variable [height, all data points flattened].
    """
    
    #Aggregate
    SCu_tower_mean, SCu_tower_std = {}, {}
    DeepCu_tower_mean, DeepCu_tower_std = {}, {}
    daily_tower = {}

    for key in data_complete:
        if key == 'time':
            continue

        SCu_mean, SCu_std = [], []
        DeepCu_mean, DeepCu_std = [], []
        daily = []

        for height in range(len(all_heights)):
            sub_SCu, sub_DeepCu, sub_daily = [], [], []

            for date_check in date_list_general:
                data = np.array(data_complete[key][date_list_general.index(date_check)][height]).astype(float)[10*2:(22*2)+1]

                if date_check in SCu_list:
                    sub_SCu.append(data)
                elif date_check in DeepCu_list:
                    sub_DeepCu.append(data)

                sub_daily.extend(data)

            SCu_mean.append(np.nanmean(sub_SCu, axis=0))
            SCu_std.append(np.nanstd(sub_SCu, axis=0))
            DeepCu_mean.append(np.nanmean(sub_DeepCu, axis=0))
            DeepCu_std.append(np.nanstd(sub_DeepCu, axis=0))
            daily.append(np.array(sub_daily))

        SCu_tower_mean[key] = np.array(SCu_mean)
        SCu_tower_std[key] = np.array(SCu_std)
        DeepCu_tower_mean[key] = np.array(DeepCu_mean)
        DeepCu_tower_std[key] = np.array(DeepCu_std)
        daily_tower[key] = np.array(daily)

    return SCu_tower_mean, SCu_tower_std, DeepCu_tower_mean, DeepCu_tower_std, daily_tower

#RADIOSONDE

#Process LCL, LFC, EL, cape, cin, and h values from sounding data for a list of dates and times
def process_values_sonde(date_list,time_list,full_path):
    
    '''
    Function to process sounding values for a list of dates.
    
    Parameters
    ----------
    date_list : list
        List of dates to process.
    full_path : str
        Full path to the CSV file containing sounding data.
    
    Returns
    -------
    values_LCL, values_LFC, values_EL, values_cape, values_cin, values_h : list
        Lists containing processed sounding values for each date.
    
    Notes
    -----
    The function reads a CSV file containing sounding data and processes it for the specified dates.
    It extracts values for LCL (Lifted Condensation Level), LFC (Level of Free Convection), EL (Equilibrium Level),
    cape (Convective Available Potential Energy), cin (Convective Inhibition), and h (height).
    The function handles specific cases, such as padding with NaN values for certain dates.
    
    If the date is 20220818, it adds two NaN values to the end of each list for missing soundings.
    The function returns lists of values for each of the sounding parameters.
    The CSV file is expected to have columns named 'Date', 'Time', 'LCL', 'LFC', 'EL', 'cape', 'cin', and 'h'.
    
    '''
    
    #Load the CSV
    df = pd.read_csv(full_path, sep=",")
    
    #Initial arrays    
    values_LCL, values_LFC, values_EL = [], [], []
    values_cape, values_cin, values_h = [], [], []

    for date in date_list:
        LCL_sub, LFC_sub, EL_sub = [], [], []
        cape_sub, cin_sub, h_sub = [], [], []

        df_sel = df.loc[df['Date'] == date]

        for time in time_list:
            df_sub = df_sel.loc[df_sel['Time'] == time]
            LCL_sub.extend(df_sub['LCL'].values)
            LFC_sub.extend(df_sub['LFC'].values)
            EL_sub.extend(df_sub['EL'].values)
            cape_sub.extend(df_sub['cape'].values)
            cin_sub.extend(df_sub['cin'].values)
            h_sub.extend(df_sub['h'].values)

        if date == 20220818:
            nan_padding = [float('nan')] * 2
            LCL_sub.extend(nan_padding)
            LFC_sub.extend(nan_padding)
            EL_sub.extend(nan_padding)
            cape_sub.extend(nan_padding)
            cin_sub.extend(nan_padding)
            h_sub.extend(nan_padding)

        values_LCL.append(LCL_sub)
        values_LFC.append(LFC_sub)
        values_EL.append(EL_sub)
        values_cape.append(cape_sub)
        values_cin.append(cin_sub)
        values_h.append(h_sub)

    return values_LCL, values_LFC, values_EL, values_cape, values_cin, values_h

# Append values from radiosonde profiles for a list of days and times
def append_values_radiosondes(day_list, time_list,category, rootdir, ref=False):    
    suffix = ' Reference' if ref else ''
    path = f'{rootdir}/{category}{suffix} Profiles/{{day}}/{{time}}UTC.csv'
    z_values, p_values, theta_values, q_values, RH_values, T_values, Td_values, wind_u_values, wind_v_values, wind_speed_values, wind_dir_values = [], [], [], [], [], [], [], [], [], [], []

    for day in day_list:
        sub_z, sub_p, sub_theta, sub_q, sub_RH, sub_T, sub_Td, sub_wind_u, sub_wind_v, sub_wind_speed, sub_wind_dir = [], [], [], [], [], [], [], [], [], [], []
        for time in time_list:
            df = pd.read_csv(path.format(day=day, time=time))
            sub_z.append(df['z'].values)
            sub_p.append(df['p'].values)
            sub_theta.append(df['theta'].values)
            sub_q.append(df['q'].values)
            sub_RH.append(df['RH'].values)
            sub_T.append(df['T'].values)
            sub_Td.append(df['Td'].values)
            sub_wind_u.append(df['wind_u'].values)
            sub_wind_v.append(df['wind_v'].values)
            sub_wind_speed.append(df['wind_speed'].values)
            sub_wind_dir.append(df['wind_dir'].values)

        z_values.append(sub_z)
        p_values.append(sub_p)
        theta_values.append(sub_theta)
        q_values.append(sub_q)
        RH_values.append(sub_RH)
        T_values.append(sub_T)
        Td_values.append(sub_Td)
        wind_u_values.append(sub_wind_u)
        wind_v_values.append(sub_wind_v)
        wind_speed_values.append(sub_wind_speed)
        wind_dir_values.append(sub_wind_dir)

    return z_values, p_values, theta_values, q_values, RH_values, T_values, Td_values, wind_u_values, wind_v_values, wind_speed_values, wind_dir_values

#Radiation 
def process_radiation_data(full_path, date_list_general, SCu_list, DeepCu_list):
    
    '''
    Function to process radiation data from a CSV file.
    
    Parameters:
    full_path : str
        The full path to the CSV file containing radiation data.
    date_list_general : list
        A list of dates for which the data needs to be processed.
    SCu_list : list
        A list of dates corresponding to the SCu period.
    DeepCu_list : list
        A list of dates corresponding to the DeepCu period.
    
    Returns:
    SCu_rad_mean : dict
        A dictionary containing the mean values of radiation for each variable during the SCu period.
    SCu_rad_std : dict
        A dictionary containing the standard deviation of radiation for each variable during the SCu period.
    DeepCu_rad_mean : dict
        A dictionary containing the mean values of radiation for each variable during the DeepCu period.
    DeepCu_rad_std : dict
        A dictionary containing the standard deviation of radiation for each variable during the DeepCu period.
    daily_rad : dict
        A dictionary containing the daily values of radiation for each variable.
    
    The keys of the dictionaries are the radiation variables, and the values are numpy arrays containing the processed data.
    The function processes the data for each date in the date_list_general, filtering the data based on the time range and calculating
    the mean and standard deviation for each variable.      
    
    '''
    
    # Load data
    df_ATTO_temp = pd.read_csv(full_path)

    # Define variables and mappings
    variables = ['SWin', 'SWout', 'LWin', 'LWout', 'T_LWin', 'T_LWout', 'Qnet', 'LWnet', 'SWnet']
    notation = {
        'SWin': 'SW_in', 'SWout': 'SW_out', 'LWin': 'LW_atm_correct', 'LWout': 'LW_terr_correct',
        'T_LWin': 'T_LW_atm', 'T_LWout': 'T_LW_terr', 'Qnet': 'Qnet', 'LWnet': 'LWnet', 'SWnet': 'SWnet'
    }
    day_rad = {var: [] for var in variables}

    # Process data for each date
    for date in date_list_general:
        df_sub = df_ATTO_temp.loc[
            (df_ATTO_temp['Date'] == date) & 
            (df_ATTO_temp['Time_correct'] >= '06:00') & 
            (df_ATTO_temp['Time_correct'] <= '18:10')
        ]
        for key in notation:
            notation_key = notation[key]
            values = df_sub[notation_key].values
            day_rad[key].append(values)

    # Aggregate statistics
    SCu_rad_mean, SCu_rad_std = {}, {}
    DeepCu_rad_mean, DeepCu_rad_std = {}, {}
    daily_rad = {}

    for key in day_rad:
        sub_SCu, sub_DeepCu, sub_daily = [], [], []

        for date_check in date_list_general:
            data = np.array(day_rad[key][date_list_general.index(date_check)]).astype(float)

            if date_check in SCu_list:
                sub_SCu.append(data)
            elif date_check in DeepCu_list:
                sub_DeepCu.append(data)

            sub_daily.extend(data)

        SCu_mean = np.nanmean(sub_SCu, axis=0) if sub_SCu else np.nan
        SCu_std = np.nanstd(sub_SCu, axis=0) if sub_SCu else np.nan
        DeepCu_mean = np.nanmean(sub_DeepCu, axis=0) if sub_DeepCu else np.nan
        DeepCu_std = np.nanstd(sub_DeepCu, axis=0) if sub_DeepCu else np.nan
        daily = np.array(sub_daily)

        SCu_rad_mean[key] = SCu_mean
        SCu_rad_std[key] = SCu_std
        DeepCu_rad_mean[key] = DeepCu_mean
        DeepCu_rad_std[key] = DeepCu_std
        daily_rad[key] = daily

    return SCu_rad_mean, SCu_rad_std, DeepCu_rad_mean, DeepCu_rad_std, daily_rad


#Clouds

# Extract cloud classification and boundary data from a grouped xarray Dataset
def extract_cloud_info(ds_mean):
    '''
    Extracts cloud classification and cloud boundary data between 06:00 and 18:00.

    Parameters:
        ds_mean (xarray.Dataset): Dataset grouped by 'hour_minute' with mean values.

    Returns:
        dict: Dictionary with arrays for classification, time, cloud base, cloud top, and cloud thickness.
    '''
    selection = ds_mean.sel(hour_minute=(ds_mean['hour_minute'] >= 600) & (ds_mean['hour_minute'] <= 1800))
    return {
        'classification_mean': selection['classification'].values,
        'time_MIRA': selection['hour_minute'].values,
        'cloud_base_mean': selection['cloud_base_mean'].values,
        'cloud_top_mean': selection['cloud_top_mean'].values,
        'cloud_thickness': selection['cloud_thickness_mean'].values
    }

#Insert NaN values at 19:00 each day into classification data
def insert_nan_at_19(total_time, total_class, nan_arr):
    '''
    Inserts a row of NaN values at 19:00 each day into classification data.

    Parameters:
        total_time (np.ndarray): Array of datetime values.
        total_class (np.ndarray): 2D array of classification data (time, z).
        nan_arr (np.ndarray): 1D array of NaN values (same shape as classification columns).

    Returns:
        tuple: Updated (total_time, total_class) with NaNs inserted after 18:00 each day.
    '''
    for i in range(len(total_time)):
        datetimer = pd.to_datetime(total_time[i])
        if datetimer.hour == 18:
            total_class = np.insert(total_class, i+1, nan_arr.reshape(1, -1), axis=0)
            total_time = np.insert(total_time, i+1, datetime(datetimer.year, datetimer.month, datetimer.day, 19, 0, 0))
    return total_time, total_class



#Respiration and GPP

#Process soil respiration data from an Excel file, resample, interpolate, and extract 30 min values
def process_soil_respiration(filepath, date_list_general, date_range):
    '''
    Reads, preprocesses, resamples, interpolates soil respiration data,
    extracts every 30 min value between 06:00–18:00 for each date,
    and returns an xarray Dataset with Resp values.

    Parameters:
        filepath (str): Path to the Excel file.
        date_list_general (list of str): List of dates ('YYYY-MM-DD') for extraction.
        date_range (pd.DatetimeIndex): Date range for full period (for creating time index).

    Returns:
        xr.Dataset: Dataset with 'Resp' variable and 'time' coordinate.
    '''
    df = pd.read_excel(filepath, sheet_name='export_1').drop([0, 1])
    df = df[df['LI-8250'] == 'C_Litter']
    df['LI-8250 UTC'] = pd.to_datetime(df['LI-8250 UTC']) - timedelta(hours=4)
    df.set_index('LI-8250 UTC', inplace=True)
    df = df[['LI-870.1']].astype(float)
    df = df.resample('1min').mean().interpolate()

    df = df.loc['2022-08-09 06:00:00':'2022-08-18 18:00:00']

    resp_values = []
    for date in date_list_general:
        daily_data = df[df.index.strftime('%Y-%m-%d') == date]
        daily_data = daily_data.between_time('06:00', '18:00')
        if not daily_data.empty:
            time_range = pd.date_range(daily_data.index.min(), daily_data.index.max(), freq='30min')
            resp_values.extend(daily_data.loc[daily_data.index.isin(time_range)]['LI-870.1'].values)
        else:
            resp_values.extend([np.nan] * len(pd.date_range('06:00', '18:00', freq='30min')))

    time_index = pd.DatetimeIndex([
        pd.Timestamp.combine(d, t) for d in date_range for t in pd.date_range('06:00', '18:00', freq='30min').time
    ])

    return xr.Dataset({'Resp': (('time'), resp_values)}, coords={'time': time_index})

#Aggregate mean and standard deviation of a variable in a dataset for specified case dates
def aggregate_case_mean_std(dataset, variable, case_list):
    '''
    Aggregates mean and standard deviation of a variable in a dataset
    for specified case dates (SCu or DeepCu) grouped by time of day.

    Parameters:
        dataset (xr.Dataset): Dataset containing the variable.
        variable (str): Name of the variable to aggregate.
        case_list (list of str): List of case dates ('YYYY-MM-DD').

    Returns:
        tuple: (mean_array, std_array) aggregated by HH.MM formatted time.
    '''
    ds_case = dataset.sel(time=dataset['time'].dt.strftime('%Y-%m-%d').isin(case_list))
    hours = ds_case.time.dt.hour
    minutes = ds_case.time.dt.minute
    ds_case = ds_case.assign_coords(time=np.array([f'{h:02d}.{m:02d}' for h, m in zip(hours, minutes)]).astype(float))
    mean = ds_case.groupby('time').mean(dim='time')[variable].values
    std = ds_case.groupby('time').std(dim='time')[variable].values
    return mean, std

#Calculate Gross Primary Production (GPP) from NEE and Respiration
def calculate_GPP_dataset(resp_dataset, NEE_sel, time_array):
    '''
    Calculates Gross Primary Production (GPP) as NEE - Respiration
    and returns it as an xarray Dataset.

    Parameters:
        resp_dataset (xr.Dataset): Dataset with 'Resp' variable.
        NEE_sel (np.ndarray): Net Ecosystem Exchange data array.
        time_array (pd.DatetimeIndex): Time coordinate for GPP Dataset.

    Returns:
        xr.Dataset: GPP Dataset with 'GPP' variable and 'time' coordinate.
    '''
    GPP_values = resp_dataset['Resp'].values - NEE_sel
    return xr.Dataset({'GPP': (('time'), GPP_values)}, coords={'time': time_array})

#Greenhouse gases
def process_ghg_data(rootdir, file, heights, date_list_general, SCu_list, DeepCu_list):
    '''
    Processes greenhouse gas data from a CSV file by filtering time, grouping by height, and aggregating by case.

    Parameters:
        rootdir (str): Directory containing the CSV file.
        file (str): Filename of the greenhouse gas data CSV.
        heights (list): List of measurement heights (as integers or floats).
        date_list_general (list of str): Dates of interest ('YYYY-MM-DD').
        SCu_list (list of str): Dates corresponding to SCu cases.
        DeepCu_list (list of str): Dates corresponding to DeepCu cases.

    Returns:
        tuple:
            SCu_mean_data (dict): Mean values for SCu case by height.
            SCu_std_data (dict): Std deviation for SCu case by height.
            DeepCu_mean_data (dict): Mean values for DeepCu case by height.
            DeepCu_std_data (dict): Std deviation for DeepCu case by height.
            daily_data (dict): Daily data flattened across all dates by height.
    '''
    df = pd.read_csv(f'{rootdir}/{file}')

    if 'CO2' in file:
        heights = heights[:]
    elif 'CH4' in file:
        heights = heights[:-1]  # Exclude highest sensor for CH4 and CO
    else:
        heights = heights[:-1]

    day_data = {f'{height}m': [] for height in heights}

    # Extract daily data by time window and date
    for date in date_list_general:
        df_sub = df[(df['Date'] == date) & (df['Time_correct'] >= '06:00:00') & (df['Time_correct'] <= '18:00:00')]
        for key in day_data:
            data = df_sub[key].values.flatten()
            day_data[key].append(np.array(data))

    # Aggregate results
    SCu_mean_data, SCu_std_data = {}, {}
    DeepCu_mean_data, DeepCu_std_data = {}, {}
    daily_data = {}

    for key, data_list in day_data.items():
        SCu_data, DeepCu_data, daily_flat = [], [], []

        for idx, date_check in enumerate(date_list_general):
            data = np.array(data_list[idx]).astype(float)
            if date_check in SCu_list:
                SCu_data.append(data)
            elif date_check in DeepCu_list:
                DeepCu_data.append(data)
            daily_flat.extend(data)

        SCu_data = np.array(SCu_data)
        DeepCu_data = np.array(DeepCu_data)

        SCu_mean_data[key] = np.nanmean(SCu_data, axis=0) if SCu_data.size > 0 else np.nan
        SCu_std_data[key] = np.nanstd(SCu_data, axis=0) if SCu_data.size > 0 else np.nan
        DeepCu_mean_data[key] = np.nanmean(DeepCu_data, axis=0) if DeepCu_data.size > 0 else np.nan
        DeepCu_std_data[key] = np.nanstd(DeepCu_data, axis=0) if DeepCu_data.size > 0 else np.nan
        daily_data[key] = np.array(daily_flat)

    return SCu_mean_data, SCu_std_data, DeepCu_mean_data, DeepCu_std_data, daily_data


# Load aircraft profile data from a text file and extract relevant variables
def load_aircraft_profile(filepath):
    '''
    Loads aircraft profile data from a text file, extracts relevant variables.

    Parameters:
        filepath (str): Path to the aircraft profile file.

    Returns:
        dict: Dictionary containing arrays for altitude (adjusted), CO2, H2O, CO, CH4.
    '''
    df = pd.read_csv(filepath).iloc[1:]

    return {
        'altitude': df['alt'].values.astype(float) - 120,
        'co2': df['co2'].values.astype(float),
        'h2o': df['h2o'].values.astype(float),
        'co': df['co'].values.astype(float),
        'ch4': df['ch4'].values.astype(float)
    }


#-----------------------------------------------
#                   D A L E S 
#-----------------------------------------------

#ABL 1D
def DALES_abl_1D(profiles):
    
    '''
    Function to calculate the ABL height from the wthvt variable in DALES profiles.
    
    Parameters:
    profiles : xarray.Dataset
        The dataset containing DALES profiles, including the wthvt variable and zt (height) variable.
        
    Returns:
    abl_1D : list
        List of ABL heights for each time reference.
    idx_1D : list
        List of indices corresponding to the ABL heights. 
    '''
    
    time_ref = profiles.time.values
    wthvt = profiles.wthvt
        
    abl_1D = []
    idx_1D = []
    for i in range(len(time_ref)):
        values = wthvt.sel(time=time_ref[i]).values
        idx = find_closest(values, np.min(values))
        
        idx = np.where(values == np.min(values))[0][0]
        abl_val = profiles.zt[idx].values+40
        
        if abl_val == 0 or abl_val > 2250 or abl_val < 500 or i<10:
            abl_val = np.nan
            idx = np.nan
        
        abl_1D.append(abl_val)
        idx_1D.append(idx)

    return abl_1D, idx_1D


#DALES profiles
def DALES_profiler(profiles):
    
    '''
    Function to extract and process DALES profiler data from the profiles dataset.
    
    Parameters:
    profiles : xarray.Dataset
        The dataset containing DALES profiler data, including variables such as time, zt, zm, thl, qt, ql, u, v, wsv001t, lwu, lwd, swu, swd, lwuca, lwdca, swuca, swdca, sv001, cfrac.
    
    Returns:
    DALES_profilers : dict
        A dictionary containing processed DALES profiler data with keys for time, zt, zm, theta, q, ql, u, v, wCO2, LWout, LWin, SWout, SWin, LWout_clear, LWin_clear, SWout_clear, SWin_clear, CO2, ac, SWnet, LWnet, Qnet, U and WD.
    The values are extracted from the profiles dataset and processed accordingly.
    Note: The zt and zm values are adjusted by adding 40 to account for the DALES model's reference height.
        
    '''

    DALES_profilers = {'time': profiles.time.values, 'zt' : profiles.zt.values+40, 'zm' : profiles.zm.values+40,
                    'theta' : profiles.thl.values, 'q' : profiles.qt.values*1000, 'ql' : profiles.ql.values,
                        'u' : profiles.u.values, 'v' : profiles.v.values, 'wCO2' : profiles.wsv001t.values/1000,
                        'LWout' : profiles.lwu.values, 'LWin' : profiles.lwd.values*-1,
                        'SWout' : profiles.swu.values, 'SWin' : profiles.swd.values*-1,
                        'LWout_clear' : profiles.lwuca.values, 'LWin_clear' : profiles.lwdca.values*-1,
                        'SWout_clear' : profiles.swuca.values, 'SWin_clear' : profiles.swdca.values*-1,
                        'CO2' : profiles.sv001.values/1000, 'ac' : profiles.cfrac.values, 
                        'SWnet' : profiles.swd.values*-1 - profiles.swu.values, 'LWnet' : profiles.lwd.values*-1 - profiles.lwu.values,
                        'Qnet' : (profiles.swd.values*-1 - profiles.swu.values) + (profiles.lwd.values*-1 - profiles.lwu.values), 'U' : np.sqrt(profiles.u.values**2 + profiles.v.values**2),
                        'WD' : (np.arctan2(-1*profiles.u.values, -1*profiles.v.values) * 180 / np.pi) % 360, 'wthetav' : profiles.wthvt.values[:,0], 'thetav' : profiles.thv.values}

    return DALES_profilers


#DALES tmser
def DALES_tmsers(tmser,tmlsm,profiles,z_correction=40):
    '''
    Function to extract and process DALES tmser data from the tmser dataset.
    
    Parameters:
    tmser : xarray.Dataset
        The dataset containing DALES tmser data, including variables such as time, cfrac, zb, zc_av, zc_max, zi, we, Qnet, H, and LE.
    tmlsm : xarray.Dataset
        The dataset containing DALES tmlsm data, including the Resp variable.
    profiles : xarray.Dataset
        The dataset containing DALES profiles, including the wsv001t variable.
    DALES_abl_1D : function
        A function to calculate the ABL height from the wthvt variable in DALES profiles.
    z_correction : int, optional
        The height correction to be applied to zb, zc_av, zc_max, and zi values. Default is 40, which accounts for the DALES model's reference height.
    
    Returns:
    DALES_tmser : dict
        A dictionary containing processed DALES tmser data with keys for time, cfrac, zb, zc_av, zc_max, zi, we, Qnet, H, and LE.
    The values are extracted from the tmser dataset and processed accordingly. The zb, zc_av, zc_max, and zi values are adjusted by adding 40 to account for the DALES model's reference height.
    
    '''

    #Open tmlsm
    headers = None
    data = {}

    count = 0
    for line in tmlsm:
        if count == 0:
            split = line.split(None)[1:]
            headers = split
            for item in headers:
                data[item] = []
            count += 1
        elif count == 1:
            count += 1
            continue
        else:
            split = line.split(None)[:]
            sub = 0
            for item in headers:
                data[item].append(split[sub])
                sub += 1
            count += 1

    DALES_Resp = (np.array(data['Resp']).astype(float)*1000)/44.009 #umol/m2/s

    wCO2_s = profiles.wsv001t.values[:,0]/1000
    wCO2_s_ds = xr.DataArray(wCO2_s,coords=[profiles.time],dims=['time'])
    wCO2_s = (wCO2_s_ds.values*1000)/44.009 #umol/m2/s

    GPP = DALES_Resp - wCO2_s
    
    #ABL
    abl_1D, idx_1D = DALES_abl_1D(profiles)   
    
    #Compose tmsers
    DALES_tmsers = {'time': tmser.time.values, 'cfrac' : tmser.cfrac.values, 'zb' : tmser.zb.values+z_correction,
                   'zc_av' : tmser.zc_av.values+z_correction, 'zc_max' : tmser.zc_max.values+z_correction,
                   'zi' : tmser.zi.values+z_correction, 'we' : tmser.we.values, 'Qnet' : tmser.Qnet.values,
                   'H' : tmser.H.values, 'LE' : tmser.LE.values, 'NEE': wCO2_s,
                   'GPP': GPP, 'abl_1D': abl_1D, 'idx_1D': idx_1D, 'EF': tmser.LE.values / (tmser.H.values + tmser.LE.values), 'Resp': DALES_Resp}
    
    tmser.close()
    tmlsm.close()
    
    return DALES_tmsers