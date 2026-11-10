'''
Scorecard for EU-CATRINE

Author: Vincent de Feiter (vincent.defeiter@wur.nl)

Version: 10 November, 2026

'''

'''
In this code we share the general code for the scorecard of EU-CATRINE based on de Feiter et al. (2026) (submitted). 
The code includes: 
> Theoretical framework function (Eq 1. de Feiter et al. (2026))
> Figure showing timeseries, relative and absolute contributions on time for the coarsest resolution, similar to de Feiter et al. (2026).
> Scorecard in tabular form for the relative and absolute contributions compared to observations and other model data across all available horizontal resolutions. 

'''

#%%
'''
#-------------------------------------------------------------------------------------------
#                           I M P O R T   L I B R A R I E S
#-------------------------------------------------------------------------------------------
'''
#Prefix
snellius_prefix = "/home/vdefeiter/"
eddy_prefix = "/home/vincent/"

prefix = snellius_prefix if os.path.exists(snellius_prefix) else eddy_prefix

#Processing
import numpy as np
import pandas as pd
from shapely import transform
import xarray as xr 
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import scipy.integrate as integrate
from scipy.signal import find_peaks
import dask
from dask.distributed import Client
from dask import delayed, compute
from scipy.interpolate import PchipInterpolator
from dask.distributed import Client, LocalCluster

#Visualisation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import matplotlib.patheffects as pe

import sys
code_utils = prefix+"EUCATRINE_WP5_6/code-utils"

if code_utils not in sys.path:
    sys.path.append(code_utils)

import testbed_functions

import importlib
importlib.reload(testbed_functions)

#%%
'''
#-------------------------------------------------------------------------------------------
#                             G E N E R A L  S E T T I N G S
#-------------------------------------------------------------------------------------------
'''
#Visualisation settings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fontsize = 20 # Define a base font size
rcParams['font.size'] = fontsize # Base font size
rcParams['axes.labelsize'] = fontsize  # Axes label font size 
rcParams['legend.fontsize'] = fontsize # Legend font size
rcParams['axes.titlepad'] = 20  # Padding between axes and title 
rcParams['axes.labelpad'] = 20  # Padding between axes and labels 
rcParams['xtick.major.pad']= 5  # Padding between ticks and labels
rcParams['ytick.major.pad']= 5  # Padding between ticks and labels

rcParams['axes.edgecolor'] = 'k'  # Color of the plot border
rcParams['axes.linewidth'] = 1  # Width of the plot border (spines)
rcParams['xtick.major.width'] = 1  # Width of major ticks on the x-axis
rcParams['ytick.major.width'] = 1  # Width of major ticks on the y-axis
rcParams['xtick.minor.width'] = 1  # Width of minor ticks on the x-axis
rcParams['ytick.minor.width'] = 1  # Width of minor ticks on the y-axis
rcParams['xtick.major.size'] = 6  # Length of major ticks on the x-axis
rcParams['ytick.major.size'] = 6  # Length of major ticks on the y-axis
rcParams['xtick.minor.size'] = 4   # Length of minor ticks on the x-axis
rcParams['ytick.minor.size'] = 4   # Length of minor ticks on the y-axis

axiswidth = 0.8 # Axis line width for gridlines

#Reference dates
SCu_list=['2022-08-09','2022-08-10','2022-08-11','2022-08-15','2022-08-17','2022-08-18']
DeepCu_list=['2022-08-12','2022-08-13','2022-08-14','2022-08-16']
date_list_general = ['2022-08-09','2022-08-10','2022-08-11','2022-08-12','2022-08-13','2022-08-14','2022-08-15','2022-08-16','2022-08-17','2022-08-18']
date_range = pd.date_range("2022-08-09","2022-08-18")

#Coordinates ATTO
x_ATTO = -59.0055
y_ATTO = -2.1458

#Specify heights
ATTO_levels = [43, 100, 127, 151, 172, 196, 223, 247, 298] #dropped 316 (not realistic)
INSTANT_levels = [5, 15, 25, 35, 50, 81]
heights_ghg = [24, 38, 53, 79, 321]

ATTO_levels = np.array(ATTO_levels)
INSTANT_levels = np.array(INSTANT_levels)

all_heights = np.sort(np.concatenate((ATTO_levels, INSTANT_levels), axis=0))

#%%
'''
#-------------------------------------------------------------------------------------------
#                                   F R A M E W O R K 
#-------------------------------------------------------------------------------------------
'''

def extended_theoretical_framework_IFS(ds_z,ds_tendency_z,ds_sfc,t_0,t_1,ds_h=None):
    '''
    Application of the Extended Theoretical Framework (ETF) to IFS data. 
    Author: Vincent S. de Feiter (vincent.defeiter@wur.nl)
    
    #Variables
    ds_z: xarray.Dataset, dataset containing the model level data with height
    ds_tendency_z: xarray.Dataset, dataset containing the tendency data
    ds_sfc: xarray.Dataset, dataset containing the surface level data
    ds_h: xarray.Dataset, dataset containing blh data
    t_0: int, starting hour of the time window for the ETF calculations
    t_1: int, ending hour of the time window for the ETF calculations (inclusive)
    
    #Output
    ds_framework: xarray.Dataset, dataset containing the results of the ETF calculations
    ds_framework_mean: xarray.Dataset, dataset containing the mean values of the ETF calculations grouped by hour
    ds_framework_std: xarray.Dataset, dataset containing the standard deviation of the ETF calculations grouped by hour
    
    '''   
    
    #Make subset of data from t_0 to t_1 per day
 
    #Tendency
    tendency_dynamic = ds_tendency_z.dco2dt_dyn.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600
    tendency_diffusive = ds_tendency_z.dco2dt_diff.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600
    tendency_convective = ds_tendency_z.dco2dt_conv.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600

    tendency_total = tendency_dynamic + tendency_diffusive + tendency_convective

    #Flux
    density = ds_tendency_z.density.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))
    flux_dynamic = ds_tendency_z.co2flx_dyn.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600
    flux_diffusive = ds_tendency_z.co2flx_diff.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600
    flux_convective = ds_tendency_z.co2flx_conv.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600

    total_flux = flux_dynamic + flux_diffusive + flux_convective

    #Mass flux - velocity
    M_down = ds_tendency_z.dmfa.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600 / density
    M_up = ds_tendency_z.umfa.sel(time=ds_tendency_z['time'].dt.hour.isin(range(t_0, t_1+1)))*3600 / density

    #Variables
    CO2 = ds_z.co2.sel(time=ds_z['time'].dt.hour.isin(range(t_0, t_1+1)))
    if ds_h is None:
        print('Using blh from original dataset')
        blh = ds_sfc.blh.sel(time=ds_sfc['time'].dt.hour.isin(range(t_0, t_1+1)))
    else:
        print('Using new blh - e.g., from gradient method')
        blh = ds_h.blh.sel(time=ds_h['time'].dt.hour.isin(range(t_0, t_1+1)))
    
    M = M_up - M_down

    #Framework

    #Variables
    C_mixed_total = []
    dC_mixed_total = []
    C_FT_total = []
    h_total = []
    dh_dt_total = []
    gamma_c_total = []
    dC_total = []
    convective_flux_at_h_total = []
    M_at_h_total = []
    integrated_tendency_dynamic_total = []
    integrated_tendency_convective_total= []
    integrated_tendency_diffusive_total = []
    integrated_tendency_total_total = []
    integrated_surface_flux_total = []
    surface_flux_diff_total = []
    C_at_h_total = []
    dC_h_dt_total = []

    #Terms
    term_1_total = []
    term_2_total = []
    term_3_total = []
    term_4_total = []
    term_5_total = []
    term_6_total = []
    term_7_total = []
    C_diagnosed_total = []

    time_total = []

    count = 0 
    for date in date_list_general: 
        
        start_datetime = pd.to_datetime(f"{date} {t_0}:00")
        end_datetime = pd.to_datetime(f"{date} {t_1}:00")
            
        sub_tendency_dynamic = tendency_dynamic.sel(time=tendency_dynamic.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        sub_tendency_convective = tendency_convective.sel(time=tendency_convective.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        sub_tendency_diffusive = tendency_diffusive.sel(time=tendency_diffusive.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))

        sub_total_tendency = tendency_total.sel(time=tendency_total.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        
        sub_flux_diffusive = flux_diffusive.sel(time=flux_diffusive.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        sub_flux_convective = flux_convective.sel(time=flux_convective.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        
        sub_CO2 = CO2.sel(time=CO2.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        sub_blh = blh.sel(time=blh.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))
        sub_M = M.sel(time=M.time.dt.strftime('%Y-%m-%d')==date).sel(time=slice(start_datetime,end_datetime))

        #Calculate per day
        C_mixed = []
        C_FT = []
        C_at_h = []
        h_values = []
        gamma_c = []
        dC_list = []
        surface_flux_diff = []
        convective_flux_at_h = []
        M_at_h = []
        integrated_tendency_dynamic = []
        integrated_tendency_convective = []
        integrated_tendency_diffusive = []
        integrated_tendency_total = []

        time_ref = sub_total_tendency.time.values
        
        count = 0
        for it in time_ref: 
            
            #Info
            blh_val = sub_blh.sel(time=it).values
            z_0 = 0  #bottom height
            
            z_sel = sub_CO2.sel(time=it).z.values
            z_sel_0_blh = sub_CO2.sel(time=it).sel(z=slice(z_0,blh_val)).z.values
                    
            #C-mixed
            CO2_sel = sub_CO2.sel(time=it).sel(z=slice(z_0,blh_val))
            C_mixed_value = CO2_sel.mean(dim='z').values
            C_mixed.append(C_mixed_value)
            h_values.append(blh_val)
            
            #Inversion depth and values at top and bottom (h)
            h_value_t0 =  sub_blh.isel(time=count).values        
            idx_h = testbed_functions.find_closest(z_sel, h_value_t0)
            dz = np.gradient(z_sel)[idx_h] #Grid spacing at h
            try:
                h_value_t1 =  sub_blh.isel(time=count+1).values
            except:
                idx_h = testbed_functions.find_closest(z_sel, h_value_t0)
                h_value_t1 = blh_val + dz
            
            delta_z = h_value_t1 - h_value_t0
            
            if delta_z < dz or delta_z<0:
                delta_z = dz
                
            z_top = blh_val + delta_z

            print("time:", it, "blh_val:", blh_val, "delta_z:", delta_z, "z_top:", z_top)
            value_at_top = sub_CO2.sel(time=it).sel(z=z_top, method='nearest').values
            value_at_h = sub_CO2.sel(time=it).sel(z=blh_val, method='nearest').values
            
            #Cft
            C_FT_value = np.nanmean(sub_CO2.sel(time=it).sel(z=slice(blh_val, z_top)).values)
            C_FT.append(C_FT_value)   
        
            #dC
            dC = C_FT_value - value_at_h
            dC_list.append(dC)
                
            #gamma
            gamma_c_value = (value_at_top - value_at_h) / (z_top - blh_val)
            gamma_c.append(float(gamma_c_value))

            #surface flux diff
            flux_diff_surface = sub_flux_diffusive.sel(time=it).sel(z=z_0, method='nearest').values
            surface_flux_diff.append(flux_diff_surface)
            
            #convective flux at h
            flux_conv_h = sub_flux_convective.sel(time=it).sel(z=blh_val, method='nearest').values
            convective_flux_at_h.append(flux_conv_h)
            
            #M at h
            M_h_value = sub_M.sel(time=it).sel(z=blh_val, method='nearest').values
            M_at_h.append(M_h_value)

            #C at h
            C_h_value = sub_CO2.sel(time=it).sel(z=blh_val, method='nearest').values
            C_at_h.append(C_h_value)
            
            #Integrated dynamic tendency
            dynamic_tendency_sel = sub_tendency_dynamic.sel(time=it).sel(z=slice(z_0,blh_val))
            
            integrate_val_dynamic_tendency = np.trapezoid(dynamic_tendency_sel.values, x=z_sel_0_blh)
            integrated_tendency_dynamic.append(integrate_val_dynamic_tendency)

            #Integrated convective tendency
            convective_tendency_sel = sub_tendency_convective.sel(time=it).sel(z=slice(z_0,blh_val))
            integrate_val_convective_tendency = np.trapezoid(convective_tendency_sel.values, x=z_sel_0_blh)
            integrated_tendency_convective.append(integrate_val_convective_tendency)
            
            #Integrated diffusive tendency
            diffusive_tendency_sel = sub_tendency_diffusive.sel(time=it).sel(z=slice(z_0,blh_val))
            integrate_val_diffusive_tendency = np.trapezoid(diffusive_tendency_sel.values, x=z_sel_0_blh)
            integrated_tendency_diffusive.append(integrate_val_diffusive_tendency)

            #Integrated total tendency
            total_tendency_sel = sub_total_tendency.sel(time=it).sel(z=slice(z_0,blh_val))
            integrate_val_total_tendency = np.trapezoid(total_tendency_sel.values, x=z_sel_0_blh)
            integrated_tendency_total.append(integrate_val_total_tendency)

            count = count + 1

        #Array
        C_mixed = np.array(C_mixed)
        C_FT = np.array(C_FT)
        h_values = np.array(h_values)
        gamma_c = np.array(gamma_c)
        dC_array = np.array(dC_list)
        surface_flux_diff = np.array(surface_flux_diff)
        convective_flux_at_h = np.array(convective_flux_at_h)
        M_at_h = np.array(M_at_h)
        integrated_tendency_diffusive = np.array(integrated_tendency_diffusive)
        integrated_tendency_convective = np.array(integrated_tendency_convective)
        integrated_tendency_dynamic = np.array(integrated_tendency_dynamic)
        integrated_tendency_total = np.array(integrated_tendency_total)
        C_at_h = np.array(C_at_h)

        #Calculate dC_mixed_dt 
        dC_mixed_dt_sub = np.diff(C_mixed).astype(float)
        dC_mixed_dt_sub = np.insert(dC_mixed_dt_sub, 0, np.nan) 

        #Calculate dh_dt
        dh_dt_sub = np.diff(h_values).astype(float)
        dh_dt_sub = np.insert(dh_dt_sub, 0, np.nan)

        #Calculate dC_h_dt
        dC_h_sub = C_mixed * h_values
        dC_h_sub = np.diff(dC_h_sub).astype(float)
        dC_h_sub = np.insert(dC_h_sub, 0, np.nan)

        #Take median of gamma_c
        median_gamma_c = np.nanmedian(gamma_c)
        gamma_c = np.array([median_gamma_c]*len(gamma_c))

        #Extend
        C_mixed_total.extend(C_mixed)
        C_FT_total.extend(C_FT)
        h_total.extend(h_values)
        gamma_c_total.extend(gamma_c)  
        dC_total.extend(dC_array)
        convective_flux_at_h_total.extend(convective_flux_at_h)
        M_at_h_total.extend(M_at_h)
        integrated_tendency_diffusive_total.extend(integrated_tendency_diffusive)   
        integrated_tendency_convective_total.extend(integrated_tendency_convective)
        integrated_tendency_dynamic_total.extend(integrated_tendency_dynamic)
        integrated_tendency_total_total.extend(integrated_tendency_total)
        dC_mixed_total.extend(dC_mixed_dt_sub)
        dh_dt_total.extend(dh_dt_sub)
        dC_h_dt_total.extend(dC_h_sub)
        C_at_h_total.extend(C_at_h)

        #Process Terms
        
        #Term 1 
        term_1 = C_FT[0]
        term_1_total.extend([term_1]*len(h_values))
        
        #Term 2
        term_2 = (C_mixed[0] - C_FT[0]) * (h_values[0] / h_values)
        term_2_total.extend(term_2)
        
        #Term 3
        time_integrated_surface_flux = np.concatenate(([0],np.nancumsum((surface_flux_diff[:-1] + surface_flux_diff[1:]) * 0.5)))
        
        term_3 = time_integrated_surface_flux / h_values
        term_3_total.extend(term_3)
        integrated_surface_flux_total.extend(time_integrated_surface_flux)
        surface_flux_diff_total.extend(surface_flux_diff)
        
        #Term 4
        term_4 = (gamma_c / (2 * h_values)) * (h_values - h_values[0])**2
        term_4_total.extend(term_4) 

        #Term 5
        time_intgrated_M = np.concatenate(([0],np.nancumsum((convective_flux_at_h[:-1] + convective_flux_at_h[1:]) * 0.5)))
    
        term_5 = time_intgrated_M / h_values
        term_5_total.extend(term_5)

        #Term 6
        M_dC = M_at_h * dC_array
        time_integrated_M_dC = np.concatenate(([0],np.nancumsum((M_dC[:-1] + M_dC[1:]) * 0.5)))
        
        term_6 = time_integrated_M_dC / h_values
        term_6_total.extend(term_6)
                
        #Term 7
        time_integrated_tendency_dynamic = np.concatenate(([0],np.nancumsum((integrated_tendency_dynamic[:-1] + integrated_tendency_dynamic[1:]) * 0.5)))
        
        term_7 = time_integrated_tendency_dynamic / h_values
        term_7_total.extend(term_7)
        
        #C Diagnosed
        C_diagnosed = term_1 + term_2 + term_3 + term_4 - term_5 + term_6 + term_7
        C_diagnosed_total.extend(C_diagnosed)

        #Time
        time_total.extend(sub_CO2.time.values)
        
        #add np.nan to all lists
        C_mixed_total.append(np.nan)
        C_FT_total.append(np.nan)
        h_total.append(np.nan)
        gamma_c_total.append(np.nan)  
        dC_total.append(np.nan)
        convective_flux_at_h_total.append(np.nan)
        M_at_h_total.append(np.nan)
        integrated_tendency_diffusive_total.append(np.nan)
        integrated_tendency_convective_total.append(np.nan)
        integrated_tendency_dynamic_total.append(np.nan)   
        integrated_tendency_total_total.append(np.nan)
        dC_mixed_total.append(np.nan)
        dh_dt_total.append(np.nan)
        term_1_total.append(np.nan) 
        term_2_total.append(np.nan)
        term_3_total.append(np.nan)
        term_4_total.append(np.nan) 
        term_5_total.append(np.nan)
        term_6_total.append(np.nan)
        term_7_total.append(np.nan)
        integrated_surface_flux_total.append(np.nan)
        C_diagnosed_total.append(np.nan) 
        surface_flux_diff_total.append(np.nan)
        dC_h_dt_total.append(np.nan)
        C_at_h_total.append(np.nan)
        
        #Time one timestep further to avoid overlap with next day
        time_total.append(sub_CO2.time.values[-1] + np.timedelta64(60, 'm'))


    #Create dataset
    ds_framework = xr.Dataset( {
        'h' : (('time'), h_total),
        'C_mixed' : (('time'), C_mixed_total),
        'C_FT' : (('time'), C_FT_total),
        'gamma_c' : (('time'), gamma_c_total),
        'dC' : (('time'), dC_total),
        'convective_flux_at_h' : (('time'), convective_flux_at_h_total),
        'surface_flux' : (('time'), surface_flux_diff_total),
        'M_at_h' : (('time'), M_at_h_total),
        'dC_mixed' : (('time'), dC_mixed_total),
        'dh_dt' : (('time'), dh_dt_total),
        'integrated_tendency_diffusive' : (('time'), integrated_tendency_diffusive_total),
        'integrated_tendency_convective' : (('time'), integrated_tendency_convective_total),
        'integrated_tendency_dynamic' : (('time'), integrated_tendency_dynamic_total),
        'integrated_tendency_total' : (('time'), integrated_tendency_total_total),
        'integrated_surface_flux' : (('time'), integrated_surface_flux_total),
        'term_1' : (('time'), term_1_total),
        'term_2' : (('time'), term_2_total),
        'term_3' : (('time'), term_3_total),
        'term_4' : (('time'), term_4_total),
        'term_5' : (('time'), term_5_total),
        'term_6' : (('time'), term_6_total),
        'term_7' : (('time'), term_7_total),
        'C_diagnosed' : (('time'), C_diagnosed_total),
        'dC_h_dt' : (('time'), dC_h_dt_total),
        'C_at_h' : (('time'), C_at_h_total),
        }, coords= { 
        'time' : time_total })

    #Averaging
    ds_framework_mean = ds_framework.groupby('time.hour').mean(dim='time')
    ds_framework_std = ds_framework.groupby('time.hour').std(dim='time')
    
    return ds_framework, ds_framework_mean, ds_framework_std


#%%
'''
#-------------------------------------------------------------------------------------------
#                           L O A D  D A T A  -  I F S
#-------------------------------------------------------------------------------------------
'''

def load_IFS_data(rootdir_IFS,EXPVER,EXPVER_tendency,YEAR,correct_hours,dates,x,y):
    
    '''
    Load IFS data. 
    Author: Vincent S. de Feiter (vincent.defeiter@wur.nl)
    
    #Variables
    rootdir_IFS: str, path to the IFS data directory
    EXPVER: str, experiment version for the IFS data with the normal data
    EXPVER_tendency: str, experiment version for the IFS tendency data
    YEAR: str, year of the data
    correct_hours: int, number of hours to correct the time axis (e.g., for timezone adjustments)
    dates: list of str, list of dates to process in 'YYYY-MM-DD' format
    x: float, longitude of the location to select (e.g., ATTO)
    y: float, latitude of the location to select (e.g., ATTO)
    
    #Output
    ds_z: xarray.Dataset, dataset containing the processed model level data
    ds_z_ghg: xarray.Dataset, dataset containing the processed model level data for heights of the ghg measured at the tower (e.g., ATTO)
    ds_tendency_z: xarray.Dataset, dataset containing the processed tendency data
    ds_sfc: xarray.Dataset, dataset containing the processed surface level data
    
    '''
        
    #Model level at z
    ds_z = xr.open_mfdataset(f'{rootdir_IFS}/{EXPVER}/ML/Processed/{EXPVER}_ml_subset_z_{YEAR}*.nc')

    #Model level at z for only ghg heights (tower)
    ds_z_ghg = xr.open_mfdataset(f'{rootdir_IFS}/{EXPVER}/ML/Processed/{EXPVER}_ml_subset_z_ghg_{YEAR}*.nc')
        
    #Tendency
    ds_tendency_z = xr.open_mfdataset(f'{rootdir_IFS}/{EXPVER_tendency}/ML/Processed/{EXPVER_tendency}_ml_subset_z_{YEAR}*.nc')

    #Surface level
    ds_sfc = xr.open_mfdataset(f'{rootdir_IFS}/{EXPVER}/SFC/Processed/{EXPVER}_sfc_subset_km_Amazon_{YEAR}*.nc')

    #Correct hours
    ds_z['time'] = pd.to_datetime(ds_z['time'].values) + pd.Timedelta(hours=correct_hours)
    ds_z_ghg['time'] = pd.to_datetime(ds_z_ghg['time'].values) + pd.Timedelta(hours=correct_hours)
    ds_tendency_z['time'] = pd.to_datetime(ds_tendency_z['time'].values) + pd.Timedelta(hours=correct_hours)
    ds_sfc['time'] = pd.to_datetime(ds_sfc['time'].values) + pd.Timedelta(hours=correct_hours)

    #Spatial mean
    ds_z = ds_z.mean(dim=['latitude', 'longitude'])
    ds_z_ghg = ds_z_ghg.mean(dim=['latitude', 'longitude'])
    ds_tendency_z = ds_tendency_z.mean(dim=['latitude', 'longitude'])
    ds_sfc = ds_sfc.mean(dim=['latitude', 'longitude'])
    
    #Select only daytime hours
    ds_z = ds_z.sel(time=ds_z['time'].dt.hour.isin(range(6, 19)))
    ds_z_ghg = ds_z_ghg.sel(time=ds_z_ghg['time'].dt.hour.isin(range(6, 19)))
    ds_tendency_z = ds_tendency_z.sel(time=ds_tendency_z['time'].dt.hour.isin(range(6, 19)))
    ds_sfc = ds_sfc.sel(time=ds_sfc['time'].dt.hour.isin(range(6, 19)))
    
    return ds_z, ds_z_ghg, ds_tendency_z, ds_sfc



#28 km
rootdir_IFS = "/gpfs/work4/0/meso/vincent/Amazon"
EXPVER = 'ia61'
EXPVER_tendency='itr2'
YEAR='2022'
correct_hours = -4
dates = date_list_general
t_0 = 8
t_1 = 16

ds_28km_z, ds_28km_z_ghg, ds_28km_tendency_z, ds_28km_sfc = load_IFS_data(rootdir_IFS,EXPVER,EXPVER_tendency,YEAR,correct_hours,dates,x_ATTO,y_ATTO)
ds_framework_28km, ds_framework_mean_28km, ds_framework_std_28km = extended_theoretical_framework_IFS(ds_28km_z, ds_28km_tendency_z, ds_28km_sfc,t_0,t_1,None)

#9 km
rootdir_IFS = "/gpfs/work4/0/meso/vincent/Amazon"
EXPVER = 'ia62'
EXPVER_tendency='iwpf'
YEAR='2022'
correct_hours = -4
dates = date_list_general
t_0 = 8
t_1 = 16

ds_9km_z, ds_9km_z_ghg, ds_9km_tendency_z, ds_9km_sfc = load_IFS_data(rootdir_IFS,EXPVER,EXPVER_tendency,YEAR,correct_hours,dates,x_ATTO,y_ATTO)
ds_framework_9km, ds_framework_mean_9km, ds_framework_std_9km = extended_theoretical_framework_IFS(ds_9km_z, ds_9km_tendency_z, ds_9km_sfc,t_0,t_1,None)

#4.4 km
rootdir_IFS = "/gpfs/work4/0/meso/vincent/Amazon"
EXPVER = 'ia63'
EXPVER_tendency='j0xh'
YEAR='2022'
correct_hours = -4
dates = date_list_general
t_0 = 8
t_1 = 16

ds_4km_z, ds_4km_z_ghg, ds_4km_tendency_z, ds_4km_sfc = load_IFS_data(rootdir_IFS,EXPVER,EXPVER_tendency,YEAR,correct_hours,dates,x_ATTO,y_ATTO)
ds_framework_4km, ds_framework_mean_4km, ds_framework_std_4km = extended_theoretical_framework_IFS(ds_4km_z, ds_4km_tendency_z, ds_4km_sfc,t_0,t_1,None)

#%%
'''
#-------------------------------------------------------------------------------------------
#                           L O A D  D A T A  -  O B S
#-------------------------------------------------------------------------------------------
'''

rootdir_OBS = prefix+"PhD/General/Reference files/"
        
#zi
df_ABL = pd.read_csv(f'{rootdir_OBS}ABL_SCu_DeepCu_aggregates.csv')
OBS_abl_time = pd.to_datetime(df_ABL['Time'].values)
OBS_abl_SCu_mean = df_ABL['SCu_mean'].values
OBS_abl_SCu_std = df_ABL['SCu_std'].values
OBS_abl_DeepCu_mean = df_ABL['DeepCu_mean'].values
OBS_abl_DeepCu_std = df_ABL['DeepCu_std'].values

df_ABL_total = pd.read_csv(f'{rootdir_OBS}ABL_filtered.csv')
abl_daily = df_ABL_total['blh'].values

#Initialize lists to store complete data
data_complete = {
    'time': [], 'T': [], 'Td': [], 'q': [], 'p': [], 'theta': [], 'TKE': [],
    'H': [], 'LE': [], 'wind_speed': [], 'max_wind_speed': [], 'wind_u': [],
    'wind_v': [], 'wind_dir': [], 'bowen_ratio': [], 'u_star': [], 'u_var': [],
    'v_var': [], 'w_var': [], 'density': [], 'cp': [], 'VPD': [], 'L': [],
    'z_d_L': [], 'wind_w': [], 'NEE': [], 'EF': []
}

#Load tower data
df_list, df_name_list = testbed_functions.load_tower_data(all_heights, rootdir_OBS)

#Process tower data
for date in date_list_general:
    day_data = testbed_functions.process_date(date, df_list)
    day_data['EF'] = np.array(day_data['LE']) / (np.array(day_data['H']) + np.array(day_data['LE'])) #add EF
    
    for key in data_complete:
        data_complete[key].append(day_data[key])

#Aggregate tower data
SCu_tower_mean, SCu_tower_std, DeepCu_tower_mean, DeepCu_tower_std, daily_tower = testbed_functions.aggregate_tower_data(
    data_complete, all_heights, date_list_general, SCu_list, DeepCu_list
)

df_050m = pd.read_csv(f'{rootdir_OBS}DF_050m.csv')

#Aircraft data
OBS_profile_1 = testbed_functions.load_aircraft_profile(f'{rootdir_OBS}Aircraft_CloudRoots_flight1_profile.txt')
OBS_profile_2 = testbed_functions.load_aircraft_profile(f'{rootdir_OBS}Aircraft_CloudRoots_flight2_profile.txt')

# Greenhouse gas processing
SCu_CO2_mean, SCu_CO2_std, DeepCu_CO2_mean, DeepCu_CO2_std, daily_CO2 = testbed_functions.process_ghg_data(rootdir_OBS, 'CloudRoots_CO2.csv', heights_ghg, date_list_general, SCu_list, DeepCu_list)
SCu_CH4_mean, SCu_CH4_std, DeepCu_CH4_mean, DeepCu_CH4_std, daily_CH4 = testbed_functions.process_ghg_data(rootdir_OBS, 'CloudRoots_CH4.csv', heights_ghg, date_list_general, SCu_list, DeepCu_list)
SCu_CO_mean, SCu_CO_std, DeepCu_CO_mean, DeepCu_CO_std, daily_CO = testbed_functions.process_ghg_data(rootdir_OBS, 'CloudRoots_CO.csv', heights_ghg, date_list_general, SCu_list, DeepCu_list)

# Group results
SCu_ghg_mean = {'CO2': SCu_CO2_mean, 'CH4': SCu_CH4_mean, 'CO': SCu_CO_mean}
SCu_ghg_std = {'CO2': SCu_CO2_std, 'CH4': SCu_CH4_std, 'CO': SCu_CO_std}
DeepCu_ghg_mean = {'CO2': DeepCu_CO2_mean, 'CH4': DeepCu_CH4_mean, 'CO': DeepCu_CO_mean}
DeepCu_ghg_std = {'CO2': DeepCu_CO2_std, 'CH4': DeepCu_CH4_std, 'CO': DeepCu_CO_std}
daily_ghg = {'CO2': daily_CO2, 'CH4': daily_CH4, 'CO': daily_CO}

#Aircraft - transects
flight1_low = pd.read_csv(f"{rootdir_OBS}/Flight1_low_detrended.csv")
flight1_low = flight1_low.iloc[1:]

flight1_low_z = flight1_low['alt'].values.astype(float)
flight1_low_co2 = flight1_low['co2'].values.astype(float)
flight1_low_LAT = flight1_low['lat'].values.astype(float)
flight1_low_LON = flight1_low['lon'].values.astype(float)
flight1_low_h2o = flight1_low['h2o'].values.astype(float)

flight1_low_z = np.nanmean(np.array(flight1_low_z)-120)
flight1_low_mean_co2 = np.nanmean(flight1_low_co2)
flight1_low_std_co2 = np.nanstd(flight1_low_co2)

flight1_low_mean_h2o = np.nanmean(flight1_low_h2o)
flight1_low_std_h2o = np.nanstd(flight1_low_h2o)

flight1_middle = pd.read_csv(f"{rootdir_OBS}/Flight1_middle_detrended.csv")
flight1_middle = flight1_middle.iloc[1:]

flight1_middle_z = flight1_middle['alt'].values.astype(float)
flight1_middle_co2 = flight1_middle['co2'].values.astype(float)
flight1_middle_LAT = flight1_middle['lat'].values.astype(float)
flight1_middle_LON = flight1_middle['lon'].values.astype(float)
flight1_middle_h2o = flight1_middle['h2o'].values.astype(float)

flight1_middle_z = np.nanmean(np.array(flight1_middle_z)-120)
flight1_middle_mean_co2 = np.nanmean(flight1_middle_co2)
flight1_middle_std_co2 = np.nanstd(flight1_middle_co2)

fligh1_middle_mean_h2o = np.nanmean(flight1_middle_h2o)
flight1_middle_std_h2o = np.nanstd(flight1_middle_h2o)

flight1_high = pd.read_csv(f"{rootdir_OBS}/Flight1_high_detrended.csv")
flight1_high = flight1_high.iloc[1:]

flight1_high_z = flight1_high['alt'].values.astype(float)
flight1_high_co2 = flight1_high['co2'].values.astype(float)
flight1_high_LAT = flight1_high['lat'].values.astype(float)
flight1_high_LON = flight1_high['lon'].values.astype(float)
flight1_high_h2o = flight1_high['h2o'].values.astype(float)

flight1_high_z = np.nanmean(np.array(flight1_high_z)-120)
flight1_high_mean_co2 = np.nanmean(flight1_high_co2)
flight1_high_std_co2 = np.nanstd(flight1_high_co2)

flight1_high_mean_h2o = np.nanmean(flight1_high_h2o)
flight1_high_std_h2o = np.nanstd(flight1_high_h2o)

flight2_low = pd.read_csv(f"{rootdir_OBS}/Flight2_low_detrended.csv")
flight2_low = flight2_low.iloc[1:]

flight2_low_z = flight2_low['alt'].values.astype(float)
flight2_low_co2 = flight2_low['co2'].values.astype(float)
flight2_low_LAT = flight2_low['lat'].values.astype(float)
flight2_low_LON = flight2_low['lon'].values.astype(float)
flight2_low_h2o = flight2_low['h2o'].values.astype(float)

flight2_low_z = np.nanmean(np.array(flight2_low_z)-120)
flight2_low_mean_co2 = np.nanmean(flight2_low_co2)
flight2_low_std_co2 = np.nanstd(flight2_low_co2)

flight2_low_mean_h2o = np.nanmean(flight2_low_h2o)
flight2_low_std_h2o = np.nanstd(flight2_low_h2o)

flight2_middle = pd.read_csv(f"{rootdir_OBS}/Flight2_middle_detrended.csv")
flight2_middle = flight2_middle.iloc[1:]

flight2_middle_z = flight2_middle['alt'].values.astype(float)
flight2_middle_co2 = flight2_middle['co2'].values.astype(float)
flight2_middle_LAT = flight2_middle['lat'].values.astype(float)
flight2_middle_LON = flight2_middle['lon'].values.astype(float)
flight2_middle_h2o = flight2_middle['h2o'].values.astype(float)

flight2_middle_z = np.nanmean(np.array(flight2_middle_z)-120)
flight2_middle_mean_co2 = np.nanmean(flight2_middle_co2)
flight2_middle_std_co2 = np.nanstd(flight2_middle_co2)

flight2_middle_mean_h2o = np.nanmean(flight2_middle_h2o)
flight2_middle_std_h2o = np.nanstd(flight2_middle_h2o)

flight2_high = pd.read_csv(f"{rootdir_OBS}/Flight2_high_detrended.csv")
flight2_high = flight2_high.iloc[1:]

flight2_high_z = flight2_high['alt'].values.astype(float)
flight2_high_co2 = flight2_high['co2'].values.astype(float)
flight2_high_LAT = flight2_high['lat'].values.astype(float)
flight2_high_LON = flight2_high['lon'].values.astype(float)
flight2_high_h2o = flight2_high['h2o'].values.astype(float)

flight2_high_z = np.nanmean(np.array(flight2_high_z)-120)
flight2_high_mean_co2 = np.nanmean(flight2_high_co2)
flight2_high_std_co2 = np.nanstd(flight2_high_co2)

flight2_high_mean_h2o = np.nanmean(flight2_high_h2o)
flight2_high_std_h2o = np.nanstd(flight2_high_h2o)

#%%
'''
#-------------------------------------------------------------------------------------------
#                           L O A D  D A T A  -  D A L E S
#-------------------------------------------------------------------------------------------
'''

#Framework DALES
rootdir_DALES = '/home/vdefeiter/CloudRoots_DALES/ShCu/Processing/Figures/'
rootdir_DALES_clouds = "/home/vdefeiter/CloudRoots_DALES/ShCu/Output/R034/"
rootdir_DALES_no_clouds = "/home/vdefeiter/CloudRoots_DALES/ShCu/Output/R035/"
rootdir_DALES_3D = "/gpfs/work3/0/lesmodels/amazon_vincent/DALES/R034_coarse/Processed/"

#=============
# 1 D  D A T A
#=============

#CLOUDS
ds_DALES_3D_clouds = xr.open_dataset(f"{rootdir_DALES}/ds_3D_clouds.nc")
ds_DALES_1D_clouds = xr.open_dataset(f"{rootdir_DALES}/ds_1D_clouds.nc")
profiles_1D_clouds = xr.open_dataset(f"{rootdir_DALES_clouds}/profiles.034.nc")
tmser_1D_clouds = xr.open_dataset(f"{rootdir_DALES_clouds}/tmser.034.nc")

x_DALES_3D_clouds = ds_DALES_3D_clouds.time.values
x_DALES_1D_clouds = ds_DALES_1D_clouds.time.values

#NO CLOUDS
ds_DALES_3D_no_clouds = xr.open_dataset(f"{rootdir_DALES}/ds_3D_no_clouds.nc")
ds_DALES_1D_no_clouds = xr.open_dataset(f"{rootdir_DALES}/ds_1D_no_clouds.nc")
profiles_1D_no_clouds = xr.open_dataset(f"{rootdir_DALES_no_clouds}/profiles.035.nc")
tmser_1D_no_clouds = xr.open_dataset(f"{rootdir_DALES_no_clouds}/tmser.035.nc")

x_DALES_3D_no_clouds = ds_DALES_3D_no_clouds.time.values
x_DALES_1D_no_clouds = ds_DALES_1D_no_clouds.time.values

ds_export = xr.open_dataset(f'/gpfs/work3/0/lesmodels/amazon_vincent/DALES/DALES_entrainment.nc')

acw_3D = ds_export['acw_3D']
acc_3D = ds_export['acc_3D']
wCO2_clouds = ds_export['wCO2_clouds']
wCO2_inclouds_acw = ds_export['wCO2_inclouds_acw']
wCO2_inclouds_acc = ds_export['wCO2_inclouds_acc']

wCO2_environment_acw = ds_export['wCO2_environment_acw']
wCO2_environment_acc = ds_export['wCO2_environment_acc']

wCO2_no_clouds = ds_export['wCO2_no_clouds']

w_inclouds_acw = ds_export['w_inclouds_acw']
w_inclouds_acc = ds_export['w_inclouds_acc']

#Reference time 1D
starttime = '2023-09-01 06:05:00'
endtime = '2023-09-01 18:00:00'
real_time_arr = pd.date_range(start=starttime, end=endtime, freq=f'300s')
real_time_arr_s = real_time_arr.strftime('%H:%M:%S')
real_time_arr =  [datetime.strptime(time_str, '%H:%M:%S').hour + datetime.strptime(time_str, '%H:%M:%S').minute / 60 for time_str in real_time_arr_s]
real_time_arr = np.array(real_time_arr)

time_1D = real_time_arr
time_ref = profiles_1D_clouds.time.values

#Reference time 3D 
if prefix == snellius_prefix:
    field_ql = xr.open_dataset('/gpfs/work3/0/lesmodels/amazon_vincent/DALES/R034_coarse/Processed/ql_processed_034.nc',chunks='auto')
else:
    field_ql = xr.open_dataset('/home/scratch2/vincent/DALES/R034/Merged/Processed/ql_processed_034.nc',chunks='auto')
time_3D = field_ql.time.values

#ABL idx 3D in 1D field
z_DALES = profiles_1D_clouds.zt.values + 40 
z_DALES_3D = acw_3D.zt.values

idx_1D_3D_no_clouds = []
idx_1D_3D_clouds = []
abl_3D_clouds = []
abl_3D_no_clouds = []

for i in range(len(time_ref)):
    
    #Select values
    values_no_clouds = profiles_1D_no_clouds.wthvt.sel(time=time_ref[i]).values
    values_clouds = profiles_1D_clouds.wthvt.sel(time=time_ref[i]).values

    #Select height with minimum value
    idx_no_clouds = np.where(values_no_clouds == np.min(values_no_clouds))[0][0]
    idx_clouds = np.where(values_clouds == np.min(values_clouds))[0][0]

    height_no_clouds = z_DALES[idx_no_clouds]
    height_clouds = z_DALES[idx_clouds]
    
    #-------------------- N O  C L O U D S --------------------
    
    #Fix after hours
    if i >= 123:
        values = profiles_1D_no_clouds.wthvt.sel(time=time_ref[123]).values
        idx = np.where(values == np.min(values))[0][0]
        height_no_clouds = z_DALES[idx]
    
    #Fix initial hours
    if height_no_clouds < 550:
        height_no_clouds = 550
        idx = testbed_functions.find_closest(z_DALES, height_no_clouds)
    
    idx_1D_3D_sub_no_clouds = testbed_functions.find_closest(z_DALES_3D, height_no_clouds)
    
    #Append
    idx_1D_3D_no_clouds.append(idx_1D_3D_sub_no_clouds)
    abl_3D_no_clouds.append(height_no_clouds)
    
    #-------------------- C L O U D S --------------------
    
    #Fix after hours
    if i >= 120:
        values = profiles_1D_clouds.wthvt.sel(time=time_ref[120]).values
        idx = np.where(values == np.min(values))[0][0]
        height_clouds = z_DALES[idx]
    
    #Fix initial hours
    if height_clouds < 550:
        height_clouds = 550
        idx = testbed_functions.find_closest(z_DALES, height_clouds)
    
    idx_1D_3D_sub_clouds = testbed_functions.find_closest(z_DALES_3D, height_clouds)
    
    #Append
    idx_1D_3D_clouds.append(idx_1D_3D_sub_clouds)
    abl_3D_clouds.append(height_clouds)

idx_1D_3D_no_clouds = np.array(idx_1D_3D_no_clouds)
idx_1D_3D_clouds = np.array(idx_1D_3D_clouds)
abl_3D_no_clouds = np.array(abl_3D_no_clouds)
abl_3D_clouds = np.array(abl_3D_clouds)

#Entrainment values at ABL height
DALES_entrainment_no_clouds = []

DALES_entrainment_in_cloud_acw = []
DALES_entrainment_in_cloud_acc = []
DALES_entrainment_environment_acw = []
DALES_entrainment_environment_acc = []
DALES_entrainment_clouds = []
DALES_M_acw = []
DALES_M_acc = []

for i in range(len(time_3D)): 
    
    #idx time
    idx_time = testbed_functions.find_closest(time_1D, time_3D[i])

    idx_abl_no_clouds = idx_1D_3D_no_clouds[idx_time]
    idx_abl_clouds = idx_1D_3D_clouds[idx_time]
    
    #-------------------- N O  C L O U D S --------------------
    
    values_no_clouds = wCO2_no_clouds.isel(time=i).isel(zt=idx_abl_no_clouds)
    DALES_entrainment_no_clouds.append(values_no_clouds)
    
    #-------------------- C L O U D S --------------------
    
    #Incloud
    values_inclouds = wCO2_inclouds_acw.isel(time=i).isel(zt=idx_abl_clouds) * acw_3D.isel(time=i).isel(zt=idx_abl_clouds)
    DALES_entrainment_in_cloud_acw.append(values_inclouds)
    
    values_inclouds = wCO2_inclouds_acc.isel(time=i).isel(zt=idx_abl_clouds) * acc_3D.isel(time=i).isel(zt=idx_abl_clouds)
    DALES_entrainment_in_cloud_acc.append(values_inclouds)
    
    #environment
    values_environment = wCO2_environment_acw.isel(time=i).isel(zt=idx_abl_clouds) * (1 - acw_3D.isel(time=i).isel(zt=idx_abl_clouds))
    DALES_entrainment_environment_acw.append(values_environment)
    
    values_environment = wCO2_environment_acc.isel(time=i).isel(zt=idx_abl_clouds) * (1 - acc_3D.isel(time=i).isel(zt=idx_abl_clouds))
    DALES_entrainment_environment_acc.append(values_environment)
    
    #total
    values_clouds = wCO2_clouds.isel(time=i).isel(zt=idx_abl_clouds)    
    DALES_entrainment_clouds.append(values_clouds)
    
    #M
    value_in_clouds = w_inclouds_acw.isel(time=i).isel(zt=idx_abl_clouds) * acw_3D.isel(time=i).isel(zt=idx_abl_clouds)
    DALES_M_acw.append(value_in_clouds)
    
    value_in_clouds = w_inclouds_acc.isel(time=i).isel(zt=idx_abl_clouds) * acc_3D.isel(time=i).isel(zt=idx_abl_clouds)
    DALES_M_acc.append(value_in_clouds)

DALES_entrainment_no_clouds = np.array(DALES_entrainment_no_clouds)
DALES_entrainment_environment_acw = np.array(DALES_entrainment_environment_acw)
DALES_entrainment_environment_acc = np.array(DALES_entrainment_environment_acc)
DALES_entrainment_clouds = np.array(DALES_entrainment_clouds)     
DALES_entrainment_in_cloud_acw = np.array(DALES_entrainment_in_cloud_acw)
DALES_entrainment_in_cloud_acc = np.array(DALES_entrainment_in_cloud_acc)
DALES_M_acw = np.array(DALES_M_acw)
DALES_M_acc = np.array(DALES_M_acc)

X = 1e-10
DALES_entrainmnt_no_clouds = np.nan_to_num(DALES_entrainment_no_clouds, nan=X)
DALES_entrainment_environment_acw = np.nan_to_num(DALES_entrainment_environment_acw, nan=X)
DALES_entrainment_environment_acc = np.nan_to_num(DALES_entrainment_environment_acc, nan=X)
DALES_entrainment_clouds = np.nan_to_num(DALES_entrainment_clouds, nan=X)     
DALES_entrainment_in_cloud_acw = np.nan_to_num(DALES_entrainment_in_cloud_acw, nan=X)
DALES_entrainment_in_cloud_acc = np.nan_to_num(DALES_entrainment_in_cloud_acc, nan=X)
DALES_M_acw = np.nan_to_num(DALES_M_acw, nan=X)
DALES_M_acc = np.nan_to_num(DALES_M_acc, nan=X)

#Make 30 min averages
ds_DALES_entrainment_values = xr.Dataset({
    'entrainment_no_clouds': (('time'), DALES_entrainment_no_clouds),
    'entrainment_clouds': (('time'), DALES_entrainment_clouds),
    'entrainment_in_cloud': (('time'), DALES_entrainment_in_cloud_acc),
    'entrainment_environment': (('time'), DALES_entrainment_environment_acc),
    'M': (('time'), DALES_M_acc)
    
}, coords={'time': pd.date_range(start="06:10", end="18:00", freq="600s")})

ds_DALES_entrainment_values_30min = ds_DALES_entrainment_values.resample(time='30min').mean()
ds_export.close()

n = len(acc_3D[:,0])
x_old = np.arange(n)*10
x_new = np.arange(0, n * 10, 5)

acc_3D_5min = np.interp(x_new, x_old, acc_3D[:,0])

#==========================
#        P I N O
#==========================

#h
time_ref = profiles_1D_clouds.time.values
z_DALES = profiles_1D_clouds.zt.values + 40 

abl_1D_clouds = []
idx_1D_clouds = []

for i in range(len(time_ref)):
    
    #Select values
    values = profiles_1D_clouds.wthvt.sel(time=time_ref[i]).values
    
    #Select height with minimum value
    idx = np.where(values == np.min(values))[0][0]

    height = z_DALES[idx]
    
    #Fix initial hours
    if i <= 23:
        height = 550
        idx = testbed_functions.find_closest(z_DALES, height)
    
    #Fix after hours
    if i >= 120:
        values = profiles_1D_clouds.wthvt.sel(time=time_ref[120]).values
        idx = np.where(values == np.min(values))[0][0]
        height = z_DALES[idx]
            
    #Append
    abl_1D_clouds.append(height)
    idx_1D_clouds.append(idx)

abl_1D_clouds = np.array(abl_1D_clouds)
idx_1D_clouds = np.array(idx_1D_clouds)

DALES_h_0_clouds = abl_1D_clouds[0]
DALES_h_clouds = abl_1D_clouds

#------------------------------------------

#CO2 mixing ratio
DALES_C_clouds = []
for i in range(len(time_ref)):
    
    #Select values
    values = profiles_1D_clouds.sv001.sel(time=time_ref[i]).values/1000  #to ppm
    
    #Cut data
    values = values[:idx_1D_clouds[i]+1]
    
    #Mean value
    mean_value = np.nanmean(values)
    
    DALES_C_clouds.append(mean_value)

DALES_C_clouds_total = np.array(DALES_C_clouds)

DALES_C_clouds = np.array(DALES_C_clouds)
DALES_C_0_clouds = DALES_C_clouds_total[0]

DALES_dC_dt_clouds = np.diff(DALES_C_clouds)/300
DALES_dC_dt_clouds = np.insert(DALES_dC_dt_clouds, 0, np.nan)  

#------------------------------------------

#w'CO2' integrated over the time period
DALES_surface_flux_clouds = (profiles_1D_clouds.wsv001t.values[:,0]/1000)  #ppm m/s

DALES_integrated_surface_flux_clouds = []
for i in range(len(DALES_surface_flux_clouds)):
    value = np.nansum(DALES_surface_flux_clouds[:i])*300
    DALES_integrated_surface_flux_clouds.append(float(value))

DALES_integrated_surface_flux_clouds = np.array(DALES_integrated_surface_flux_clouds)

#------------------------------------------

#CO2 lapse rate and FT
DALES_dC_clouds = []
DALES_C_at_h_clouds = []
DALES_C_FT_clouds = []
DALES_gamma_c_clouds = []
DALES_delta_z_clouds = []

DALES_dh_dt_clouds = np.gradient(DALES_h_clouds)

z_cloud_top = tmser_1D_clouds.zc_av.values+40

for i in range(len(time_ref)):
    
    #i = 50
    
    #Select values
    values = profiles_1D_clouds.sv001.sel(time=time_ref[i]).values/1000  #to ppm
    flux_theta_v = profiles_1D_clouds.wthvt.sel(time=time_ref[i]).values
    z_LES = profiles_1D_clouds.zt.values + 40
    
    zero_line = [-1e-3] * len(z_LES)
    zero_line = np.array(zero_line)

    h_value = abl_1D_clouds[i]
    h_idx = testbed_functions.find_closest(z_LES, h_value)

    #cut data
    flux_theta_v = flux_theta_v[h_idx:]
    z_LES = z_LES[h_idx:]
    zero_line = zero_line[h_idx:]
    
    f = flux_theta_v - zero_line

    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]

    if len(idx) == 0:
        z_cross = np.nan
    else:
        a = idx[0]
        z_cross = z_LES[a] - f[a] * (z_LES[a+1] - z_LES[a]) / (f[a+1] - f[a])

    z_cross = float(z_cross)
    delta_z = z_cross - h_value
        
    #Fix initial hours
    if i < 23:
        delta_z = 15
   
    #Fix after hours
    if i >= 123:
        delta_z =  50
    
    z_top = h_value + delta_z
    
    print(i,delta_z)
       
    idx_top = testbed_functions.find_closest(z_DALES, z_top)
    idx_h = testbed_functions.find_closest(z_DALES, h_value)
    
    #Gamma_c 
    value_at_top = values[idx_top]
    value_at_h = values[idx_h]
    
    gamma_c = (value_at_top - value_at_h) / (z_top - h_value)
                
    #C_FT
    C_FT = np.nanmean(values[idx_h:idx_top+1])
    
    #dC
    dC = C_FT - value_at_h
        
    #Append
    DALES_dC_clouds.append(float(dC))
    DALES_C_at_h_clouds.append(values[idx_h])
    DALES_C_FT_clouds.append(C_FT)
    DALES_gamma_c_clouds.append(gamma_c)
    DALES_delta_z_clouds.append(delta_z)
    


#Array
DALES_dC_clouds = np.array(DALES_dC_clouds)
DALES_C_at_h_clouds = np.array(DALES_C_at_h_clouds)
DALES_C_FT_clouds = np.array(DALES_C_FT_clouds)
DALES_gamma_c_clouds = np.array(DALES_gamma_c_clouds)

#Make constant
DALES_gamma_c_clouds = [np.nanmedian(DALES_gamma_c_clouds)]*len(DALES_gamma_c_clouds)

DALES_delta_z_clouds = np.array(DALES_delta_z_clouds)

#==========================
#  C A L C U L A T I O N  
#==========================

#Pino et al.
DALES_term_1_clouds_Pino = [DALES_C_FT_clouds[0]]*len(DALES_h_clouds)
DALES_term_2_clouds_Pino = (DALES_C_0_clouds - DALES_C_FT_clouds[0]) * (DALES_h_0_clouds / DALES_h_clouds)
DALES_term_3_clouds_Pino = DALES_integrated_surface_flux_clouds / DALES_h_clouds
DALES_term_4_clouds_Pino = ((DALES_gamma_c_clouds) / (2 * DALES_h_clouds)) * (DALES_h_clouds - DALES_h_0_clouds)**2

DALES_C_Pino_clouds = DALES_term_1_clouds_Pino + DALES_term_2_clouds_Pino + DALES_term_3_clouds_Pino + DALES_term_4_clouds_Pino

#de Feiter et al. (30 min)
DALES_term_1_clouds = [DALES_C_FT_clouds[0]]*len(DALES_h_clouds)
DALES_term_1_clouds = xr.Dataset({'term_1_clouds': (('time'), DALES_term_1_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().term_1_clouds
DALES_term_2_clouds = (DALES_C_0_clouds - DALES_C_FT_clouds[0]) * (DALES_h_0_clouds / DALES_h_clouds)
DALES_term_2_clouds = xr.Dataset({'term_2_clouds': (('time'), DALES_term_2_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().term_2_clouds
DALES_term_3_clouds = DALES_integrated_surface_flux_clouds / DALES_h_clouds
DALES_term_3_clouds = xr.Dataset({'term_3_clouds': (('time'), DALES_term_3_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().term_3_clouds
DALES_term_4_clouds = ((DALES_gamma_c_clouds) / (2 * DALES_h_clouds)) * (DALES_h_clouds - DALES_h_0_clouds)**2
DALES_term_4_clouds = xr.Dataset({'term_4_clouds': (('time'), DALES_term_4_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().term_4_clouds


#Mass flux term (clouds) - term 5
DALES_dC_clouds_30min = xr.Dataset({'dC_clouds': (('time'), DALES_dC_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().dC_clouds

integrated_M_dC = []
for i in range(len(ds_DALES_entrainment_values_30min.entrainment_in_cloud)):
    value = np.nansum(ds_DALES_entrainment_values_30min.M[:i]*DALES_dC_clouds_30min[:i])*(60*30)
    integrated_M_dC.append(float(value))
integrated_M_dC = np.array(integrated_M_dC)

DALES_h_clouds_30min = xr.Dataset({'h_clouds': (('time'), DALES_h_clouds)}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")}).resample(time='30min').mean().h_clouds

DALES_term_5_clouds = integrated_M_dC/DALES_h_clouds_30min

#Mass flux term (30 min) - term 6
integrated_M = []
for i in range(len(ds_DALES_entrainment_values_30min.entrainment_in_cloud)):
    value = np.nansum(ds_DALES_entrainment_values_30min.entrainment_in_cloud[:i])*(60*30)
    integrated_M.append(float(value))
integrated_M = np.array(integrated_M)


DALES_term_6_clouds = integrated_M/DALES_h_clouds_30min 

DALES_C_diagnosed_clouds = DALES_term_1_clouds + DALES_term_2_clouds + DALES_term_3_clouds + DALES_term_4_clouds + DALES_term_5_clouds - DALES_term_6_clouds


time_DALES = pd.date_range(start="06:05", end="18:00", freq="300s")
idx_8LT = int(np.where((time_DALES.hour == 8) & (time_DALES.minute == 0))[0][0])
idx_16LT = int(np.where((time_DALES.hour == 16) & (time_DALES.minute == 0))[0][0])

#To 30 min dataset
ds_DALES_clouds = xr.Dataset({
    'h': (('time'), DALES_h_clouds),
    'C_mixed': (('time'), DALES_C_clouds),
    'dC': (('time'), DALES_dC_clouds),
    'dC_dt': (('time'), DALES_dC_dt_clouds),
    'surface_flux': (('time'), DALES_surface_flux_clouds),
    'integrated_surface_flux': (('time'), DALES_integrated_surface_flux_clouds),
    'gamma_c': (('time'), DALES_gamma_c_clouds),
    'C_FT': (('time'), DALES_C_FT_clouds),
    'C_0': (('time'), np.full_like(DALES_h_clouds, DALES_C_0_clouds)),
    'h_0': (('time'), np.full_like(DALES_h_clouds, DALES_h_0_clouds)),
    'term_1_Pino': (('time'), DALES_term_1_clouds_Pino),
    'term_2_Pino': (('time'), DALES_term_2_clouds_Pino),
    'term_3_Pino': (('time'), DALES_term_3_clouds_Pino),
    'term_4_Pino': (('time'), DALES_term_4_clouds_Pino),
    'C_Pino': (('time'), DALES_C_Pino_clouds),
    'delta_z': (('time'), DALES_delta_z_clouds),
    'term_7': (('time'), np.array([0]*len(DALES_h_clouds)))  # Placeholder for term_7,
}, coords={'time': pd.date_range(start="06:05", end="18:00", freq="300s")})

ds_DALES_clouds_30min = ds_DALES_clouds.resample(time='30min').mean()

#Append diagnosed terms and final result to ds_DALES_clouds_30min
ds_DALES_clouds_30min['term_1'] = DALES_term_1_clouds
ds_DALES_clouds_30min['term_2'] = DALES_term_2_clouds
ds_DALES_clouds_30min['term_3'] = DALES_term_3_clouds
ds_DALES_clouds_30min['term_4'] = DALES_term_4_clouds
ds_DALES_clouds_30min['term_5'] = DALES_term_5_clouds
ds_DALES_clouds_30min['term_6'] = DALES_term_6_clouds
ds_DALES_clouds_30min['C_diagnosed'] = DALES_C_diagnosed_clouds
ds_DALES_clouds_30min['M_flux'] = ds_DALES_entrainment_values_30min.entrainment_in_cloud
ds_DALES_clouds_30min['M'] = ds_DALES_entrainment_values_30min.M

plt.plot(ds_DALES_clouds_30min.C_mixed,color='tab:gray')
plt.plot(ds_DALES_clouds_30min.C_Pino)
plt.plot(ds_DALES_clouds_30min.C_diagnosed,linestyle='--')

plt.show()

'''
#=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 
#=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
'''

#%%
'''
#----------------------------------------------------------------------------------------------
 V I S U A L I S A T I O N - T I M E  S E R I E S - D I U R N A L  E V O L U T I O N (M E A N)
#----------------------------------------------------------------------------------------------
'''

fig, ax = plt.subplots(2,1,figsize=(11.69/1.2,11.69/1.2),dpi=300) #figsize = width, height in inches (A4 size)
ax1,ax2 = ax.flatten()

#===================================================================
#             P L O T  A  - D I U R N A L  C O 2
#===================================================================

#--------------------------
# O B S E R V A T I O N S
#--------------------------

rootdir = rootdir_OBS
file = 'CloudRoots_CO2.csv'
df = pd.read_csv(f'{rootdir}/{file}')

heights = heights_ghg

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

OBS_total_24m = day_data['24m']
OBS_total_38m = day_data['38m']
OBS_total_53m = day_data['53m']
OBS_total_79m = day_data['79m']
OBS_total_321m = day_data['321m']

OBS_total_24m_mean = np.nanmean(OBS_total_24m, axis=0)
OBS_total_38m_mean = np.nanmean(OBS_total_38m, axis=0)
OBS_total_53m_mean = np.nanmean(OBS_total_53m, axis=0)
OBS_total_79m_mean = np.nanmean(OBS_total_79m, axis=0)
OBS_total_321m_mean = np.nanmean(OBS_total_321m, axis=0)

OBS_total_24m_std = np.nanstd(OBS_total_24m, axis=0)
OBS_total_38m_std = np.nanstd(OBS_total_38m, axis=0)
OBS_total_53m_std = np.nanstd(OBS_total_53m, axis=0)
OBS_total_79m_std = np.nanstd(OBS_total_79m, axis=0)
OBS_total_321m_std = np.nanstd(OBS_total_321m, axis=0)

#Pandas time range from 06:00 to 18:00 with 1min frequency
x_OBS = pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="1min")

#Visualisation
colors = ['tab:gray','tab:blue','tab:orange','#D8B40C','tab:red']
p = 0
for height in [24, 38, 53, 79, 321]:
    height_str = f'{height}m'

    ax1.plot(x_OBS,locals()[f'OBS_total_{height_str}_mean'],linewidth=axiswidth*2,color=colors[p],ls="",marker='o',markersize=8,markeredgecolor='k',markeredgewidth=0.5)
    ax1.fill_between(x_OBS,locals()[f'OBS_total_{height_str}_mean']-locals()[f'OBS_total_{height_str}_std'],locals()[f'OBS_total_{height_str}_mean']+locals()[f'OBS_total_{height_str}_std'],color=colors[p],alpha=0.5)

    p += 1

#--------------------------
#   E C M W F  -  I F S 
#--------------------------

IFS_28km_mean = ds_framework_mean_28km
IFS_28km_std = ds_framework_std_28km

CO2_28km = ds_28km_z.co2

# Mean boundary-layer height for each time step
blh_28km = ds_28km_sfc.blh

C_mixed_total = []

for i, blh in enumerate(blh_28km.values):
    CO2_sel_blh = (
        CO2_28km
        .isel(time=i)
        .sel(z=slice(0, float(blh)))
        .mean(dim="z")
    )

    C_mixed_total.append(CO2_sel_blh)

dataset_C_mixed = xr.Dataset(
    {
        "C_mixed": ("time", C_mixed_total)
    },
    coords={
        "time": ds_28km_z.time
    }
)

CO2 = ds_28km_z_ghg.co2

ECMWF_28km_24m_total = []
ECMWF_28km_38m_total = []
ECMWF_28km_53m_total = []
ECMWF_28km_79m_total = []
ECMWF_28km_321m_total = []

for date in date_list_general: 
        
    sub_CO2 = CO2.sel(time=date)
    
    ECMWF_28km_24m_total.append(sub_CO2.sel(z=24, method='nearest').values)
    ECMWF_28km_38m_total.append(sub_CO2.sel(z=38, method='nearest').values)
    ECMWF_28km_53m_total.append(sub_CO2.sel(z=53, method='nearest').values)
    ECMWF_28km_79m_total.append(sub_CO2.sel(z=79, method='nearest').values)
    ECMWF_28km_321m_total.append(sub_CO2.sel(z=321, method='nearest').values)

ECWMF_28km_24m_total = np.array(ECMWF_28km_24m_total)
ECMWF_28km_38m_total = np.array(ECMWF_28km_38m_total)
ECMWF_28km_53m_total = np.array(ECMWF_28km_53m_total)
ECMWF_28km_79m_total = np.array(ECMWF_28km_79m_total)
ECMWF_28km_321m_total = np.array(ECMWF_28km_321m_total)

#Mean
ECMWF_28km_24m_mean = np.nanmean(ECMWF_28km_24m_total, axis=0)
ECMWF_28km_38m_mean = np.nanmean(ECMWF_28km_38m_total, axis=0)
ECMWF_28km_53m_mean = np.nanmean(ECMWF_28km_53m_total, axis=0)     
ECMWF_28km_79m_mean = np.nanmean(ECMWF_28km_79m_total, axis=0)
ECMWF_28km_321m_mean = np.nanmean(ECMWF_28km_321m_total, axis=0)

#STD
ECMWF_28km_24m_std = np.nanstd(ECMWF_28km_24m_total, axis=0)
ECMWF_28km_38m_std = np.nanstd(ECMWF_28km_38m_total, axis=0)
ECMWF_28km_53m_std = np.nanstd(ECMWF_28km_53m_total, axis=0)     
ECMWF_28km_79m_std = np.nanstd(ECMWF_28km_79m_total, axis=0)
ECMWF_28km_321m_std = np.nanstd(ECMWF_28km_321m_total, axis=0)

#IFS
x_IFS = pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="60min")
x_IFS_c = pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="30min")

ECMWF_28km_24m_mean_c = np.full(len(ECMWF_28km_24m_mean) * 2, np.nan)
ECMWF_28km_24m_mean_c[::2] = ECMWF_28km_24m_mean
ECMWF_28km_24m_mean_c = ECMWF_28km_24m_mean_c[:-1]

ECMWF_28km_38m_mean_c = np.full(len(ECMWF_28km_38m_mean) * 2, np.nan)
ECMWF_28km_38m_mean_c[::2] = ECMWF_28km_38m_mean
ECMWF_28km_38m_mean_c = ECMWF_28km_38m_mean_c[:-1]

ECMWF_28km_53m_mean_c = np.full(len(ECMWF_28km_53m_mean) * 2, np.nan)
ECMWF_28km_53m_mean_c[::2] = ECMWF_28km_53m_mean
ECMWF_28km_53m_mean_c = ECMWF_28km_53m_mean_c[:-1]

ECMWF_28km_79m_mean_c = np.full(len(ECMWF_28km_79m_mean) * 2, np.nan)
ECMWF_28km_79m_mean_c[::2] = ECMWF_28km_79m_mean
ECMWF_28km_79m_mean_c = ECMWF_28km_79m_mean_c[:-1]

ECMWF_28km_321m_mean_c = np.full(len(ECMWF_28km_321m_mean) * 2, np.nan)
ECMWF_28km_321m_mean_c[::2] = ECMWF_28km_321m_mean
ECMWF_28km_321m_mean_c = ECMWF_28km_321m_mean_c[:-1]

ECMWF_28km_24m_std_c = np.full(len(ECMWF_28km_24m_std) * 2, np.nan)
ECMWF_28km_24m_std_c[::2] = ECMWF_28km_24m_std
ECMWF_28km_24m_std_c = ECMWF_28km_24m_std_c[:-1]

ECMWF_28km_38m_std_c = np.full(len(ECMWF_28km_38m_std) * 2, np.nan)
ECMWF_28km_38m_std_c[::2] = ECMWF_28km_38m_std
ECMWF_28km_38m_std_c = ECMWF_28km_38m_std_c[:-1]

ECMWF_28km_53m_std_c = np.full(len(ECMWF_28km_53m_std) * 2, np.nan)
ECMWF_28km_53m_std_c[::2] = ECMWF_28km_53m_std
ECMWF_28km_53m_std_c = ECMWF_28km_53m_std_c[:-1]

ECMWF_28km_79m_std_c = np.full(len(ECMWF_28km_79m_std) * 2, np.nan)
ECMWF_28km_79m_std_c[::2] = ECMWF_28km_79m_std
ECMWF_28km_79m_std_c = ECMWF_28km_79m_std_c[:-1]

ECMWF_28km_321m_std_c = np.full(len(ECMWF_28km_321m_std) * 2, np.nan)
ECMWF_28km_321m_std_c[::2] = ECMWF_28km_321m_std    
ECMWF_28km_321m_std_c = ECMWF_28km_321m_std_c[:-1]

#Diagnosed and direct C_mixed
x_IFS_framework = pd.date_range(start="2024-01-01 08:00", end="2024-01-01 17:00", freq="60min")
x_IFS_framework_c = pd.date_range(start="2024-01-01 08:00", end="2024-01-01 17:00", freq="30min")

y_IFS_mean_C_mixed = IFS_28km_mean.C_mixed
y_IFS_std_C_mixed = IFS_28km_std.C_mixed
y_IFS_mean_C_diagnosed = IFS_28km_mean.C_diagnosed
y_IFS_std_C_diagnosed = IFS_28km_std.C_diagnosed

y_IFS_mean_C_mixed_c = np.full(len(y_IFS_mean_C_mixed) * 2, np.nan)
y_IFS_mean_C_mixed_c[::2] = y_IFS_mean_C_mixed
y_IFS_mean_C_mixed_c = y_IFS_mean_C_mixed_c[:-1]

y_IFS_std_C_mixed_c = np.full(len(y_IFS_std_C_mixed) * 2, np.nan)
y_IFS_std_C_mixed_c[::2] = y_IFS_std_C_mixed
y_IFS_std_C_mixed_c = y_IFS_std_C_mixed_c[:-1]

y_IFS_mean_C_diagnosed_c = np.full(len(y_IFS_mean_C_diagnosed) * 2, np.nan)
y_IFS_mean_C_diagnosed_c[::2] = y_IFS_mean_C_diagnosed
y_IFS_mean_C_diagnosed_c = y_IFS_mean_C_diagnosed_c[:-1]

y_IFS_std_C_diagnosed_c = np.full(len(y_IFS_std_C_diagnosed) * 2, np.nan)
y_IFS_std_C_diagnosed_c[::2] = y_IFS_std_C_diagnosed
y_IFS_std_C_diagnosed_c = y_IFS_std_C_diagnosed_c[:-1]

#Visualisation
colors = ['tab:gray','tab:blue','tab:orange','#D8B40C','tab:red']
p = 0
for height in [24, 38, 53, 79, 321]:
    height_str = f'{height}m'

    ax1.plot(x_IFS,locals()[f'ECMWF_28km_{height_str}_mean'],markersize=8,linewidth=axiswidth*3,color=colors[p],marker='s',markeredgecolor='k',markeredgewidth=0.5,ls="-")
    ax1.fill_between(x_IFS_c,locals()[f'ECMWF_28km_{height_str}_mean_c']-locals()[f'ECMWF_28km_{height_str}_std_c'],locals()[f'ECMWF_28km_{height_str}_mean_c']+locals()[f'ECMWF_28km_{height_str}_std_c'],color=colors[p],alpha=0.3)

    p += 1

#--------------------------
#      D A L E S 
#--------------------------
colors_2 = ['tab:orange','#D8B40C','tab:red']
p = 0
x_DALES = pd.date_range(start="2024-01-01 06:05", end="2024-01-01 18:00", freq="300s")
for z in [53, 79, 321]:

    idx = testbed_functions.find_closest(z_DALES, z)
    values = profiles_1D_clouds.sv001[:,idx].values/1000

    #Select data per 30 min
    values_30min = []
    for i in range(0,len(values),6):
        mean_value = np.nanmean(values[i:i+6])
        values_30min.append(mean_value)
    values_30min = np.array(values_30min)

    x_30min = pd.date_range(start="2024-01-01 06:05", end="2024-01-01 18:00", freq="30min")

    ax1.plot(x_30min, values_30min, linewidth=axiswidth*3, color=colors_2[p], ls="-", marker='o', markersize=8, markeredgecolor='k', markeredgewidth=0.5)
    p = p + 1

#Detailing
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='major')
ax1.tick_params(which='minor')

ax1.grid(linestyle=':', alpha=0.8,color='tab:gray')

ax1.set_xlabel('LT (hour)',labelpad=10)
ax1.set_ylabel('CO$_2$ (ppm)')
ax1.set_ylim(400,500)

#Format xaxis to show hours every 2 hours
ax1.set_xticks(pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="2h"),labels=[f'{hour}' for hour in range(6, 19, 2)])
ax1.set_xlim(pd.Timestamp("2024-01-01 05:45"), pd.Timestamp("2024-01-01 18:15"))

#===================================================================
#        P L O T  B  - F R A M E W O R K  F I T  C O 2
#===================================================================

#--------------------------
#       O B S 
#--------------------------

ax2.plot(x_OBS,OBS_total_53m_mean,linewidth=axiswidth*2,color='tab:gray',ls="",marker='o',markersize=8,markeredgecolor='k',markeredgewidth=0.5)
ax2.fill_between(x_OBS,OBS_total_53m_mean-OBS_total_53m_std,OBS_total_53m_mean+OBS_total_53m_std,color='tab:gray',alpha=0.5)

#--------------------------
#   E C M W F  -  I F S 
#--------------------------
ax2.plot(x_IFS_framework, y_IFS_mean_C_mixed, linestyle='-', color='darkgreen',markersize=8, linewidth=axiswidth*3, marker='s', markeredgecolor='k', markeredgewidth=0.5)
ax2.fill_between(x_IFS_framework_c, y_IFS_mean_C_mixed_c - y_IFS_std_C_mixed_c, y_IFS_mean_C_mixed_c + y_IFS_std_C_mixed_c, color='tab:gray', alpha=0.3)

ax2.plot(x_IFS_framework, y_IFS_mean_C_diagnosed, linestyle='-', color='k', markersize=8, linewidth=axiswidth*3, marker='s', markeredgecolor='k', markeredgewidth=0.5)
ax2.fill_between(x_IFS_framework_c, y_IFS_mean_C_diagnosed_c - y_IFS_std_C_diagnosed_c, y_IFS_mean_C_diagnosed_c + y_IFS_std_C_diagnosed_c, color='k', alpha=0.3)

#--------------------------
#      D A L E S 
#--------------------------

x_DALES = pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="30min")
ax2.plot(x_DALES,ds_DALES_clouds_30min.C_mixed, linestyle='-', color='darkgreen',markersize=8, linewidth=axiswidth*3, marker='o', markeredgecolor='k', markeredgewidth=0.5)
ax2.plot(x_DALES,ds_DALES_clouds_30min.C_diagnosed, linestyle='-', color='k',markersize=8, linewidth=axiswidth*3, marker='o', markeredgecolor='k', markeredgewidth=0.5)

#Detailing
ax2.tick_params(axis='x')
ax2.tick_params(axis='y')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='major')
ax2.tick_params(which='minor')

ax2.grid(linestyle=':', alpha=0.8,color='tab:gray')

ax2.set_xlabel('LT (hour)',labelpad=10)
ax2.set_ylabel('CO$_2$ (ppm)')
ax2.set_ylim(410,480)

#Format xaxis to show hours every 2 hours
ax2.set_xticks(pd.date_range(start="2024-01-01 06:00", end="2024-01-01 18:00", freq="2h"),labels=[f'{hour}' for hour in range(6, 19, 2)])
ax2.set_xlim(pd.Timestamp("2024-01-01 05:45"), pd.Timestamp("2024-01-01 18:15"))

plt.subplots_adjust(hspace=0.3)

ax1.text(-0.2, 0.95, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=plt.rcParams['font.size'], fontweight='bold', va='top', ha='left')
ax2.text(-0.2, 0.95, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=plt.rcParams['font.size'], fontweight='bold', va='top', ha='left')


legend_elements = [
    Patch(facecolor=colors[0], edgecolor='k', label='24 m', linewidth=0.5),
    Patch(facecolor=colors[1], edgecolor='k', label='38 m', linewidth=0.5),
    Patch(facecolor=colors[2], edgecolor='k', label='53 m', linewidth=0.5),
    Patch(facecolor=colors[3], edgecolor='k', label='79 m', linewidth=0.5),
    Patch(facecolor=colors[4], edgecolor='k', label='321 m', linewidth=0.5),
    Patch(facecolor='none', linewidth=0.5),
    
    Line2D(
        [0], [0],
        color='k',
        marker='o',
        markersize=8,
        markeredgewidth=0.5,
        markeredgecolor='k',
        markerfacecolor='w',
        linestyle='None',
        linewidth=axiswidth*2,
        label='OBS'
    ),

    Line2D(
        [0], [0],
        color='k',
        linestyle='-',
        markersize=8,
        marker='s',
        markerfacecolor='w',
        linewidth=axiswidth*3,
        label='IFS (28 km)'
    ),

    Line2D(
        [0], [0],
        color='k',
        linestyle='-',
        markersize=8,
        marker='o',
        markerfacecolor='w',
        linewidth=axiswidth*3,
        label='DALES'
    ),

]

ax1.legend(
    handles=legend_elements,
    loc='upper left',                 
    bbox_to_anchor=(-0.12, 1.47),         
    fontsize=rcParams['font.size'],
    frameon=False,
    framealpha=1,
    borderpad=1,
    handlelength=1.5,
    handleheight=1.5,
    borderaxespad=0.5,
    labelspacing=0.5,
    columnspacing=1,
    ncol=5,
    title_fontsize=rcParams['font.size'] * 0.9
)


legend_2 = [Line2D([0], [0],
        color='darkgreen',
        linestyle='-',
        markersize=8,
        marker='s',
        markerfacecolor='darkgreen',
        linewidth=axiswidth*3,
        label=r'$\langle \mathrm{C} \rangle_{\mathrm{model}}$'
    ),
    
    Line2D([0], [0],
        color='k',
        linestyle='-',
        markersize=8,
        marker='s',
        markerfacecolor='k',
        linewidth=axiswidth*3,
        label=r'$\langle \mathrm{C} \rangle_{\mathrm{diagnosed}}$'
    ),
    Line2D(
            [0], [0],
            color='tab:gray',
            marker='o',
            markersize=8,
            markeredgewidth=0.5,
            markeredgecolor='k',
            markerfacecolor='tab:gray',
            linestyle='None',
            linewidth=axiswidth*2,
            label='OBS (53 m)'
        ),
    ]

ax2.legend(
    handles=legend_2,
    loc='upper left',                 
    bbox_to_anchor=(0.55, 0.9),         
    fontsize=rcParams['font.size'],
    frameon=False,
    framealpha=1,
    borderpad=1,
    handlelength=1.5,
    handleheight=1.5,
    borderaxespad=0.5,
    labelspacing=0.5,
    columnspacing=1,
    ncol=1,
    title_fontsize=rcParams['font.size'] * 0.9
)


#Export
outdir = '/home/vdefeiter/EUCATRINE_WP5_6/Results'

if not os.path.exists(f"{outdir}/scorecard/"):
    os.makedirs(f"{outdir}/scorecard/")

plt.savefig(f"{outdir}/scorecard/Timeseries.pdf",bbox_inches='tight')   
plt.savefig(f"{outdir}/scorecard/Timeseries.jpg",bbox_inches='tight',dpi=300)

plt.show()
plt.close()
#%%
'''
#--------------------------------------------------------------------------------------------
#                           A T T R I B U T I O N S
#--------------------------------------------------------------------------------------------
'''

def relative_and_absolute_contributions(ds_framework):
    
    '''
    Function to calculate the relative and absolute contributions of each term in the extended theoretical framework.
    
    Parameters:
    ds_framework: xarray dataset containing the extended theoretical framework results
    
    Returns:
    ds_abs_contrib: xarray dataset containing the absolute contributions of each term
    ds_rel_contrib: xarray dataset containing the relative contributions of each term
        
    
    '''
        
    t = ds_framework.time

    df = pd.DataFrame({
        #"term_1": ds_framework.term_1.sel(time='2022-08-09'),
        "term_2": ds_framework.term_2,
        "term_3": ds_framework.term_3,
        "term_4": ds_framework.term_4,
        "term_5": ds_framework.term_5*-1,  
        "term_6": ds_framework.term_6,
        'term_7': ds_framework.term_7,
        'R': ds_framework.C_diagnosed - ds_framework.C_mixed 
    }, index=t)
    
    contributions = df.copy()

    #Absolute contributions (incl. direction)
    abs_contrib = contributions
    
    #Relative contributions (absolute value)
    rel_contrib = contributions.abs().div(contributions.abs().sum(axis=1), axis=0) * 100 #relative share per row in percentage

    #To xarray dataset 
    ds_abs_contrib = abs_contrib.to_xarray()
    ds_rel_contrib = rel_contrib.to_xarray()
    
    return ds_abs_contrib, ds_rel_contrib


fig, ax = plt.subplots(3, 1, figsize=(8.27,11.69),dpi=800) #figsize = width, height in inches
ax1,ax2,ax3 = ax.flatten()

#left,bottom,width,height
ax1.set_position([0.1, 0.7, 0.8, 0.25])
ax2.set_position([1.02, 0.7, 0.2, 0.25])
ax3.set_position([0.1, 0.05, 1.12, 0.4])

#===================================================================
#       P L O T  - A & B |   A B S.  C O N T R I B U T I O N S
#===================================================================

colors = {
    "term_1": "k",  # green
    "term_2": "#17becf",  # cyan
    "term_3": "#1f77b4",  # blue
    "term_4": "#9467bd",  # purple
    "term_5": "#d62728",  # red
    "term_6": "#e377c2",  # pink/magenta
    "term_7": "#ff7f0e",  # orange
    'R': 'tab:gray'
}

labels_list = [r'$(\langle C \rangle_0 - C_0^{FT})\frac{h_0}{h}$',
            r"$\frac{\left[ \overline{w'c'}_s \right]}{h}$",
            r"$\frac{\gamma_c}{2h} (h - h_0)^2$",
            r"$-\frac{\left[ \overline{w'c'}_M \right]}{h}$",
            r"$\frac{\left[ M \Delta C \right]}{h}$",
            r"$\frac{\left[ \int_{s}^{h} \frac{\partial \overline{C}}{\partial t}\bigg|_{dyn} \: dz \right]}{h}$"]

terms = ["term_2", "term_3", "term_4", "term_5", "term_6", "term_7", "R"]

#Obtain absolute and relative contributions
abs_contrib, rel_contrib = relative_and_absolute_contributions(ds_framework_28km)

abs_contrib_mean = abs_contrib.groupby('index.hour').mean(dim='index')
abs_contrib_std = abs_contrib.groupby('index.hour').std(dim='index')

rel_contrib_mean = rel_contrib.groupby('index.hour').mean(dim='index')
rel_contrib_std = rel_contrib.groupby('index.hour').std(dim='index')

#Corrected <C>
C_mixed_term_1 = ds_framework_28km.C_mixed - ds_framework_28km.term_1

C_mixed_term_1_mean = C_mixed_term_1.groupby('time.hour').mean(dim='time')
C_mixed_term_1_std = C_mixed_term_1.groupby('time.hour').std(dim='time')

#Term 1
IFS_28km_mean = ds_framework_mean_28km
IFS_28km_std = ds_framework_std_28km

term_1_mean = np.nanmean(ds_framework_28km.term_1.values)
term_1_std = np.nanstd(ds_framework_28km.term_1.values)

n_groups = len(IFS_28km_mean.hour)
n_bars = len(terms)

group_gap = 1.5         # increase for more space between hour groups
bar_width = 0.2

x = np.arange(n_groups) * group_gap 
offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

#Visualise bars
ax1.bar(x+offsets[0], abs_contrib_mean.term_2.values, width=bar_width, color=colors["term_2"], edgecolor="k", label=labels_list[0], linewidth=0.6,yerr=abs_contrib_std.term_2.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[1], abs_contrib_mean.term_3.values, width=bar_width, color=colors["term_3"], edgecolor="k", label=labels_list[1], linewidth=0.6,yerr=abs_contrib_std.term_3.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[2], abs_contrib_mean.term_4.values, width=bar_width, color=colors["term_4"], edgecolor="k", label=labels_list[2], linewidth=0.6,yerr=abs_contrib_std.term_4.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[3], abs_contrib_mean.term_5.values, width=bar_width, color=colors["term_5"], edgecolor="k", label=labels_list[3], linewidth=0.6,yerr=abs_contrib_std.term_5.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[4], abs_contrib_mean.term_6.values, width=bar_width, color=colors["term_6"], edgecolor="k", label=labels_list[4], linewidth=0.6,yerr=abs_contrib_std.term_6.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[5], abs_contrib_mean.term_7.values, width=bar_width, color=colors["term_7"], edgecolor="k", label=labels_list[5], linewidth=0.6,yerr=abs_contrib_std.term_7.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)
ax1.bar(x+offsets[6], abs_contrib_mean.R.values, width=bar_width, color=colors["R"], edgecolor="k", label="R", linewidth=0.6,yerr=abs_contrib_std.R.values, capsize=2, error_kw={"ecolor": "k","elinewidth": 0.3},zorder=2)

# Vertical separators between groups
for xpos in (x[:-1] + x[1:]) / 2:
    ax1.axvline(xpos, color="0.7", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

# Above each tick, add the top of the plot add C_mixed_term_1_mean +- C_mixed_term_1_std as text
for i in range(len(x)-1):
    mean_val = C_mixed_term_1_mean.values[i]
    std_val = C_mixed_term_1_std.values[i]

    # Mean (bold)
    ax1.text(
        x[i],
        11.6,
        rf"$\mathbf{{{mean_val:.1f}}}$",
        ha='center',
        va='bottom',
        fontsize=rcParams['font.size'] * 0.8,
    )

    # Standard deviation (smaller)
    ax1.text(
        x[i],
        11.45,
        rf"$\pm\,{std_val:.1f}$",
        ha='center',
        va='top',
        fontsize=rcParams['font.size'] * 0.6,
    )

ax1.axhline(0, color="k", linewidth=0.8)

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

ax1.yaxis.set_minor_locator(AutoMinorLocator())

ax1.grid(axis="y", linestyle=":", color='tab:gray', alpha=0.8)
ax1.grid(axis="x", visible=False)

ax1.set_xlabel("LT (hour)", labelpad=10)
ax1.set_ylabel("(ppm)", labelpad=10)

ax1.set_xticks(x)
ax1.set_xticklabels([f"{hour}" for hour in IFS_28km_mean.hour.values])

for i in range(len(x)):
    if i % 2 == 0:  # shade every other group
        ax1.axvspan(
            x[i] - group_gap/2,
            x[i] + group_gap/2,
            color='grey',
            alpha=0.08,
            zorder=0
        )
        
ax1.set_xlim(x[0] - 0.9, x[-2] + 0.9)
ax1.set_ylim(-10, 10)
# ax1.set_yticks(np.arange(-10, 25, 5),np.arange(-10, 25, 5).astype(int))


#----------------------------------------------------------------------------

#Visualise term 1
ax2.errorbar(1, term_1_mean, yerr=term_1_std, color='k', marker='s', linewidth=0.6, linestyle='-',capsize=5, zorder=2, ecolor='k')

#Detailing
ax2.set_xticks([1])
ax2.set_xticklabels([r'$C_0^{FT}$'])
ax2.set_xlim(0.5, 1.5)
ax2.set_yticks(np.arange(410, 431, 5),np.arange(410, 431, 5).astype(int))
ax2.set_ylim(413.5, 430)

ax2.tick_params(axis='x')
ax2.tick_params(axis='y')

ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='major')
ax2.tick_params(which='minor')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.set_ylabel('(ppm)',labelpad=10)

ax2.grid(linestyle=':', color='tab:gray', alpha=0.8)

#----------------------------------------------------------------------------
#             P L O T  - C  |   R E L.  C O N T R I B U T I O N S
#----------------------------------------------------------------------------

#Visualise
x = pd.date_range(start="2024-01-01 08:00", end="2024-01-01 17:00", freq="60min")
x = x.hour

bottom = np.zeros(len(rel_contrib_mean.hour))
for col in rel_contrib_mean.data_vars:
    try:
        ax3.bar(
            x,
            rel_contrib_mean[col].values,
            bottom=bottom,
            width=0.85, #0.85,
            label=labels_list[int(col.split('_')[1]) - 2],  # adjust index based on term number
            color=colors.get(col, None),   # fallback if missing
            edgecolor="none"            
        )
        bottom += rel_contrib_mean[col].values
    except:
        ax3.bar(
            x,
            rel_contrib_mean[col].values,
            bottom=bottom,
            width=0.85, #0.85,
            label='R', # adjust index based on term number
            color=colors.get(col, None),   # fallback if missing
            edgecolor="none"            
        )
        bottom += rel_contrib_mean[col].values
    
    #Add value in the bar
    for i in range(len(x)):
        v = rel_contrib_mean[col].values[i]
        v_2 = rel_contrib_std[col].values[i]

        v_pct = v 
        v2_pct = v_2 

        # Do not show std if value is 100
        v_txt = f"{v_pct:.1f}" if v_pct < 100 else "100"
        v2_txt = f"{v2_pct:.1f}" if v_pct < 100 else ""
        
        if v_txt == "100":
            show_std = False
        else:
            show_std = True

        if v > 0.08*100:

            if show_std:
                # Large bars → stacked labels

                ax3.text(
                    x[i],
                    bottom[i] - v/2 + 0.02*100,
                    rf"$\mathbf{{{v_txt}}}$",
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['font.size'] * 0.75,
                    color='white',
                )

                ax3.text(
                    x[i],
                    bottom[i] - v/2 - 0.02*100,
                    rf"$\pm {v2_txt}$",
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['font.size'] * 0.6,
                    color='white',
                )

            else:
                ax3.text(
                    x[i],
                    bottom[i] - v/2,
                    rf"$\mathbf{{{v_txt}}}$",
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['font.size'] * 0.75,
                    color='white',
                )

        elif v > 0.05*100:

            if show_std:
                # Medium bars → inline labels

                ax3.text(
                    x[i],
                    bottom[i] - v/2,
                    rf"$\mathbf{{{v_txt}}} \pm {v2_txt}$",
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['font.size'] * 0.55,
                    color='white',
                )

            else:
                ax3.text(
                    x[i],
                    bottom[i] - v/2,
                    rf"$\mathbf{{{v_txt}}}$",
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['font.size'] * 0.55,
                    color='white',
                )

#Detailing
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.xaxis.set_minor_locator(AutoMinorLocator())

ax3.set_ylim(0, 100)
ax3.set_yticks(np.arange(0,101,20), (np.arange(0,101,20)).astype(int))
ax3.set_ylabel('(\%)',labelpad=10)

ax3.grid(linestyle=':',color='tab:gray',alpha=0.3,linewidth=axiswidth*2)
ax3.set_xlabel('LT (hour)',labelpad=10)

#Major ticks every 2 hours, minor ticks every hour
ax3.set_xticks(np.arange(8, 18, 2))
ax3.set_xticks(np.arange(8, 18, 1), minor=True)

ax3.set_xlim(7.5, 16.5)

ax3.legend(loc='lower left', bbox_to_anchor= (-0.02,1.05), ncol=4,
                borderaxespad=0, frameon=False,fontsize=plt.rcParams['font.size'])

#---------------

#Labels all axes
ax1.text(-0.15, 1.05, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=rcParams['font.size'], fontweight='bold', va='top', ha='right')
ax2.text(-0.02, 1.2, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=rcParams['font.size'], fontweight='bold', va='top', ha='right')
ax3.text(-0.11, 1.05, r'\textbf{(c)}', transform=ax3.transAxes, fontsize=rcParams['font.size'], fontweight='bold', va='top', ha='right')    

#Export
outdir = '/home/vdefeiter/EUCATRINE_WP5_6/Results'

if not os.path.exists(f"{outdir}/scorecard/"):
    os.makedirs(f"{outdir}/scorecard/")

plt.savefig(f"{outdir}/scorecard/Contributions.pdf",bbox_inches='tight')   
plt.savefig(f"{outdir}/scorecard/Contributions.jpg",bbox_inches='tight',dpi=300)

plt.show()
plt.close()


#%%
'''
#--------------------------------------------------------------------------------------------
#                         S C O R E  C A R D  -  T A B L E
#--------------------------------------------------------------------------------------------
'''

def error_to_obs_and_residual(ds_framework):
    
    #Error
    error = []
    for i in range(len(date_list_general)): 
        
        OBS_sel = pd.DataFrame({'CO2':OBS_total_53m[i]}, index=pd.date_range(start=f"{date_list_general[i]} 06:00", end=f"{date_list_general[i]} 18:00", freq="1min"))
        OBS_sel_60min = OBS_sel.resample('60min').mean().interpolate()
        OBS_sel_60min = OBS_sel_60min.loc[f"{date_list_general[i]} {t_0}:00":f"{date_list_general[i]} {t_1+1}:00"]
        
        OBS_ref = OBS_sel_60min['CO2'].values
        
        sub_error = np.abs(ds_framework.C_mixed.sel(time=date_list_general[i]) - OBS_ref)
        error.append(sub_error.values)

    error = np.array(error)

    error_mean = np.nanmean(error, axis=0)
    error_std = np.nanstd(error, axis=0)

    #Residual
    residual = ds_framework.C_mixed - ds_framework.C_diagnosed

    residual_mean = residual.groupby('time.hour').mean(dim='time')
    residual_std = residual.groupby('time.hour').std(dim='time')
    
    return error, error_mean, error_std, residual, residual_mean, residual_std


#Calculate relative and absolute contributions for the framework datasets
ds_28km_abs_contrib, ds_28km_rel_contrib = relative_and_absolute_contributions(ds_framework_28km)
ds_9km_abs_contrib, ds_9km_rel_contrib = relative_and_absolute_contributions(ds_framework_9km)
ds_4km_abs_contrib, ds_4km_rel_contrib = relative_and_absolute_contributions(ds_framework_4km)

#Group by hours and take mean 
ds_28km_abs_contrib_mean = ds_28km_abs_contrib.sel(index=ds_28km_abs_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()
ds_9km_abs_contrib_mean = ds_9km_abs_contrib.sel(index=ds_9km_abs_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()
ds_4km_abs_contrib_mean = ds_4km_abs_contrib.sel(index=ds_4km_abs_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()

ds_28km_rel_contrib_mean = ds_28km_rel_contrib.sel(index=ds_28km_rel_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()
ds_9km_rel_contrib_mean = ds_9km_rel_contrib.sel(index=ds_9km_rel_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()
ds_4km_rel_contrib_mean = ds_4km_rel_contrib.sel(index=ds_4km_rel_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean() 

#Differences 9 - 28 km
diff_rel_contrib_9_28km = ds_9km_rel_contrib - ds_28km_rel_contrib
diff_rel_contrib_9_28km_mean = diff_rel_contrib_9_28km.sel(index=ds_9km_rel_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()

diff_rel_contrib_4_28km = ds_4km_rel_contrib - ds_28km_rel_contrib
diff_rel_contrib_4_28km_mean = diff_rel_contrib_4_28km.sel(index=ds_4km_rel_contrib['index'].dt.strftime('%Y-%m-%d').isin(SCu_list)).groupby('index.hour').mean()

#DALES to hours
ds_DALES_hour = ds_DALES_clouds_30min.resample(time='1h').mean()
ds_DALES_hour = ds_DALES_hour.sel(time=ds_DALES_hour.time.dt.hour.isin(range(8,17)))
ds_DALES_abs_contrib, ds_DALES_rel_contrib = relative_and_absolute_contributions(ds_DALES_hour)

#Time DALES to hours
ds_DALES_abs_contrib['index'] = ds_DALES_abs_contrib.index.dt.hour
ds_DALES_abs_contrib = ds_DALES_abs_contrib.rename({'index': 'hour'})

ds_DALES_rel_contrib['index'] = ds_DALES_rel_contrib.index.dt.hour
ds_DALES_rel_contrib = ds_DALES_rel_contrib.rename({'index': 'hour'})

#Residual and init 
error_28km, error_mean_28km, error_std_28km, residual_28km, residual_mean_28km, residual_std_28km = error_to_obs_and_residual(ds_framework_28km)
error_9km, error_mean_9km, error_std_9km, residual_9km, residual_mean_9km, residual_std_9km = error_to_obs_and_residual(ds_framework_9km)
error_4km, error_mean_4km, error_std_4km, residual_4km, residual_mean_4km, residual_std_4km = error_to_obs_and_residual(ds_framework_4km)

error_DALES = []
for i in range(len(SCu_list)): 
    
    OBS_sel = pd.DataFrame({'CO2':OBS_total_53m[i]}, index=pd.date_range(start=f"{SCu_list[i]} 06:00", end=f"{SCu_list[i]} 18:00", freq="1min"))
    OBS_sel_60min = OBS_sel.resample('60min').mean().interpolate()
    OBS_sel_60min = OBS_sel_60min.loc[f"{SCu_list[i]} {t_0}:00":f"{SCu_list[i]} {t_1}:00"]
    
    OBS_ref = OBS_sel_60min['CO2'].values
    
    sub_error = np.abs(ds_DALES_hour['C_mixed'] - OBS_ref)
    error_DALES.append(sub_error.values)

error_DALES = np.array(error_DALES)
error_DALES_mean = np.nanmean(error_DALES, axis=0)

#Add to ds_DALES_rel_contrib
ds_DALES_rel_contrib['error'] = (('hour'), error_DALES_mean)

#Scorecard
sel_hour = 12
idx_hour = np.where(ds_28km_rel_contrib_mean.hour == sel_hour)[0][0]
df = pd.DataFrame({
r'$\textbf{Term}$': [r'\textbf{II}',r'\textbf{III}',r'\textbf{IV}',r'\textbf{V}',r'\textbf{VI}',r'\textbf{VII}',r'\textbf{R}',r'\textbf{$\epsilon$}'],
    r'$\textbf{Physical Meaning}$':['Mixed-Layer Growth Suppression',
                     'Ecosystem Exchange (plant \n assimilation and soil respiration)',
                     'Dry Convective Mixing (determined \n by the CO$_2$ lapse rate)',
                     'Moist Convective Dilution Suppression',
                     'Moist Convective Direct Ventilation',
                     'Large-Scale Advection',
                     'Framework Residual',
                     'Initial Error'],
    r"$\textbf{DALES}$": [np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).term_2.values, 2), np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).term_3.values, 2), 
                   np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).term_4.values, 2), np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).term_5.values, 2), 
                   np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).term_6.values, 2), '-',
                   np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).R.values, 2), np.round(ds_DALES_rel_contrib.sel(hour=sel_hour).error.values, 2)],
    r"$\textbf{ECMWF 28 km}$": [np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_2.values, 2), np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_3.values, 2), 
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_4.values, 2), np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_5.values, 2), 
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_6.values, 2), np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_7.values, 2),
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).R.values, 2), np.round(error_mean_28km[idx_hour], 2)],
    r"$\textbf{ECMWF 9 km}$": [np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_2.values, 2), np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_3.values, 2), 
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_4.values, 2), np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_5.values, 2), 
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_6.values, 2), np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_7.values, 2),
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).R.values, 2), np.round(error_mean_9km[idx_hour], 2)],
    r"$\textbf{ECMWF 4 km}$": [np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_2.values, 2), np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_3.values, 2), 
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_4.values, 2), np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_5.values, 2), 
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_6.values, 2), np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_7.values, 2),
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).R.values, 2), np.round(error_mean_4km[idx_hour], 2)],
    r'$\Delta \mathrm{28km} - \mathrm{DALES}$': [np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_2.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_2.values, 2), np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_3.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_3.values, 2),
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_4.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_4.values, 2), np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_5.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_5.values, 2),
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).term_6.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_6.values, 2), '-',
                   np.round(ds_28km_rel_contrib_mean.sel(hour=sel_hour).R.values - ds_DALES_rel_contrib.sel(hour=sel_hour).R.values, 2), 
                   np.round(error_mean_28km[idx_hour] - ds_DALES_rel_contrib.sel(hour=sel_hour).error.values, 2)],
    r'$\Delta \mathrm{9km} - \mathrm{DALES}$': [np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_2.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_2.values, 2), np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_3.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_3.values, 2),
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_4.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_4.values, 2), np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_5.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_5.values, 2),
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).term_6.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_6.values, 2), '-',
                   np.round(ds_9km_rel_contrib_mean.sel(hour=sel_hour).R.values - ds_DALES_rel_contrib.sel(hour=sel_hour).R.values, 2), 
                   np.round(error_mean_9km[idx_hour] - ds_DALES_rel_contrib.sel(hour=sel_hour).error.values, 2)],
    r'$\Delta \mathrm{4km} - \mathrm{DALES}$': [np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_2.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_2.values, 2), np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_3.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_3.values, 2),
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_4.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_4.values, 2), np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_5.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_5.values, 2),
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).term_6.values - ds_DALES_rel_contrib.sel(hour=sel_hour).term_6.values, 2), '-',
                   np.round(ds_4km_rel_contrib_mean.sel(hour=sel_hour).R.values - ds_DALES_rel_contrib.sel(hour=sel_hour).R.values, 2), 
                   np.round(error_mean_4km[idx_hour] - ds_DALES_rel_contrib.sel(hour=sel_hour).error.values, 2)],
    
    # r'$\Delta 9 - 28 km$': [np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_2.values, 2), np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_3.values, 2),
    #                np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_4.values, 2), np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_5.values, 2),
    #                np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_6.values, 2), np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).term_7.values, 2),
    #                np.round(diff_rel_contrib_9_28km_mean.sel(hour=sel_hour).R.values, 2), '-'],
    # r'$\Delta 4 - 28 km$': [np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_2.values, 2), np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_3.values, 2),
    #                np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_4.values, 2), np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_5.values, 2),
    #                np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_6.values, 2), np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).term_7.values, 2),
    #                np.round(diff_rel_contrib_4_28km_mean.sel(hour=sel_hour).R.values, 2), '-']
})


fig, ax = plt.subplots(figsize=(11.69/1.5,8.27/2),dpi=300) #
col_widths = [0.08,0.44, 0.12, 0.2, 0.2, 0.2,0.2, 0.2, 0.2]
ax.axis('off')  # hide axes

table = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   loc='center',
                 cellLoc='left',
                    colWidths=col_widths)


#Adjust font size
table.auto_set_font_size(False)
table.set_fontsize(10)

#Scale table (key for row height!)
table.scale(1, 1.6)  # (x_scale, y_scale)

#Colour
target_cols = [r'$\Delta \mathrm{28km} - \mathrm{DALES}$', r'$\Delta \mathrm{9km} - \mathrm{DALES}$', r'$\Delta \mathrm{4km} - \mathrm{DALES}$']

# Red for large magnitude, green near 0
cmap = mcolors.LinearSegmentedColormap.from_list(
    "red_green_red",
    ["#d73027", "#1a9850", "#d73027"]
)

# Red for large magnitude, green near 0
cmap = mcolors.LinearSegmentedColormap.from_list(
    "red_green_red",
    ["#d73027", "#1a9850", "#d73027"]
)
for col_name in target_cols:
    try:
        col_idx = df.columns.get_loc(col_name)
        values = pd.to_numeric(df[col_name], errors="coerce")

        max_abs = values.abs().max()

        if pd.isna(max_abs) or max_abs == 0:
            continue

        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

        for row in range(len(df)):
            cell = table[(row + 1, col_idx)]
            value = pd.to_numeric(df.iloc[row, col_idx], errors="coerce")

            if pd.isna(value):
                cell.set_facecolor("lightgrey")
                cell.get_text().set_color("black")
                continue

            color = cmap(norm(value))
            cell.set_facecolor(color)

            r, g, b, _ = color
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            cell.get_text().set_color("white" if brightness < 0.5 else "black")

    except Exception as e:
        print(f"Error for {col_name}: {e}")
        
#Set row height
for (row, col), cell in table.get_celld().items():
    h = 0.12
    cell.set_height(h)

ax.set_title(f'ScoreCard ShCu Regime - CloudRoots-Amazon22 - {sel_hour} LT', fontsize=14, fontweight='bold', pad=30)

#Export
outdir = '/home/vdefeiter/EUCATRINE_WP5_6/Results'

if not os.path.exists(f"{outdir}/scorecard/"):
    os.makedirs(f"{outdir}/scorecard/")

plt.savefig(f"{outdir}/scorecard/Scorecard_ShCu_{sel_hour}LT.pdf",bbox_inches='tight')   
plt.savefig(f"{outdir}/scorecard/Scorecard_ShCu_{sel_hour}LT.jpg",bbox_inches='tight',dpi=300)

plt.show()
plt.close()


