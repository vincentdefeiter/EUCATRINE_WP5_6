"""
IFS Processing Functions

Author: Vincent de Feiter. Contact: vincent.defeiter@wur.nl

Version: 19 October, 2026

"""

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   I M P O R T   L I B R A R I E S 
#------------------------------------------------------------------------------------------------------------------------------------------------------
from matplotlib import units
from matplotlib import units
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # additional plotting functionalities
import numpy as np
import xarray as xr
import pandas as pd
import os
import re
import scipy
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from scipy.interpolate import PchipInterpolator
import glob
import dask
from dask.diagnostics import ProgressBar
import geopandas as gpd  # reading shapefiles
import regionmask  # used for region mapping


#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                              V A R I A B L E   D I C T I O N A R I E S    
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Model level
dict_model = {
    "210061": {"variable_name": "Carbon dioxide mass mixing ratio", "shortname": "co2", "units": "kg kg-1", "cumulative": False},
    "129": {"variable_name": "Geopotential", "shortname": "z", "units": "m", "cumulative": False},
    "246": {"variable_name": "Specific cloud liquid water content", "shortname": "clwc", "units": "kg kg-1", "cumulative": False},
    "133": {"variable_name": "Specific humidity", "shortname": "q", "units": "kg kg-1", "cumulative": False},
    "75": {"variable_name": "Specific rain water content", "shortname": "crwc", "units": "kg kg-1", "cumulative": False},
    "130": {"variable_name": "Temperature", "shortname": "t", "units": "K", "cumulative": False},
    "162107": {"variable_name": "Time-integrated downdraught detrainment rate", "shortname": "ddra", "units": "kg m-3", "cumulative": True},
    "162105": {"variable_name": "Time-integrated downdraught mass flux", "shortname": "dmfa", "units": "kg m-2", "cumulative": True},
    "162101": {"variable_name": "Time-integrated temperature tendency due to long-wave radiation", "shortname": "trta", "units": "K", "cumulative": True},
    "162100": {"variable_name": "Time-integrated temperature tendency due to short-wave radiation", "shortname": "srta", "units": "K", "cumulative": True},
    "162109": {"variable_name": "Time-integrated turbulent diffusion coefficient for heat", "shortname": "tdcha", "units": "m2", "cumulative": True},
    "162106": {"variable_name": "Time-integrated updraught detrainment rate", "shortname": "udra", "units": "kg m-3", "cumulative": True},
    "162104": {"variable_name": "Time-integrated updraught mass flux", "shortname": "umfa", "units": "kg m-2", "cumulative": True},
    "131": {"variable_name": "U component of wind", "shortname": "u", "units": "m s-1", "cumulative": False},
    "132": {"variable_name": "V component of wind", "shortname": "v", "units": "m s-1", "cumulative": False},
    "135": {"variable_name": "Vertical velocity", "shortname": "w", "units": "Pa s-1", "cumulative": False},
    "138": {"variable_name": "Vorticity (relative)", "shortname": "vo", "units": "s-1", "cumulative": False},
    
    # dynamics
    'p212001': {'variable_name': 'dudt_dyn',   'shortname': 'dudt_dyn',   'units': '-', 'cumulative': True},
    'p212002': {'variable_name': 'uflx_dyn',   'shortname': 'uflx_dyn',   'units': '-', 'cumulative': True},
    'p212003': {'variable_name': 'dvdt_dyn',   'shortname': 'dvdt_dyn',   'units': '-', 'cumulative': True},
    'p212004': {'variable_name': 'vflx_dyn',   'shortname': 'vflx_dyn',   'units': '-', 'cumulative': True},
    'p212005': {'variable_name': 'dTdt_dyn',   'shortname': 'dTdt_dyn',   'units': '-', 'cumulative': True},
    'p212006': {'variable_name': 'Tflx_dyn',   'shortname': 'Tflx_dyn',   'units': '-', 'cumulative': True},
    'p212007': {'variable_name': 'dqdt_dyn',   'shortname': 'dqdt_dyn',   'units': '-', 'cumulative': True},
    'p212008': {'variable_name': 'qflx_dyn',   'shortname': 'qflx_dyn',   'units': '-', 'cumulative': True},
    'p212009': {'variable_name': 'dco2dt_dyn', 'shortname': 'dco2dt_dyn', 'units': '-', 'cumulative': True},
    'p212010': {'variable_name': 'co2flx_dyn', 'shortname': 'co2flx_dyn', 'units': '-', 'cumulative': True},
    'p212011': {'variable_name': 'dch4dt_dyn', 'shortname': 'dch4dt_dyn', 'units': '-', 'cumulative': True},
    'p212012': {'variable_name': 'ch4flx_dyn', 'shortname': 'ch4flx_dyn', 'units': '-', 'cumulative': True},

    # diffusion
    'p212013': {'variable_name': 'dudt_diff',   'shortname': 'dudt_diff',   'units': '-', 'cumulative': True},
    'p212014': {'variable_name': 'uflx_diff',   'shortname': 'uflx_diff',   'units': '-', 'cumulative': True},
    'p212015': {'variable_name': 'dvdt_diff',   'shortname': 'dvdt_diff',   'units': '-', 'cumulative': True},
    'p212016': {'variable_name': 'vflx_diff',   'shortname': 'vflx_diff',   'units': '-', 'cumulative': True},
    'p212017': {'variable_name': 'dTdt_diff',   'shortname': 'dTdt_diff',   'units': '-', 'cumulative': True},
    'p212018': {'variable_name': 'Tflx_diff',   'shortname': 'Tflx_diff',   'units': '-', 'cumulative': True},
    'p212019': {'variable_name': 'dqdt_diff',   'shortname': 'dqdt_diff',   'units': '-', 'cumulative': True},
    'p212020': {'variable_name': 'qflx_diff',   'shortname': 'qflx_diff',   'units': '-', 'cumulative': True},
    'p212021': {'variable_name': 'dco2dt_diff', 'shortname': 'dco2dt_diff', 'units': '-', 'cumulative': True},
    'p212022': {'variable_name': 'co2flx_diff', 'shortname': 'co2flx_diff', 'units': '-', 'cumulative': True},
    'p212023': {'variable_name': 'dch4dt_diff', 'shortname': 'dch4dt_diff', 'units': '-', 'cumulative': True},
    'p212024': {'variable_name': 'ch4flx_diff', 'shortname': 'ch4flx_diff', 'units': '-', 'cumulative': True},

    # convection
    'p212025': {'variable_name': 'dudt_conv',   'shortname': 'dudt_conv',   'units': '-', 'cumulative': True},
    'p212026': {'variable_name': 'uflx_conv',   'shortname': 'uflx_conv',   'units': '-', 'cumulative': True},
    'p212027': {'variable_name': 'dvdt_conv',   'shortname': 'dvdt_conv',   'units': '-', 'cumulative': True},
    'p212028': {'variable_name': 'vflx_conv',   'shortname': 'vflx_conv',   'units': '-', 'cumulative': True},
    'p212029': {'variable_name': 'dTdt_conv',   'shortname': 'dTdt_conv',   'units': '-', 'cumulative': True},
    'p212030': {'variable_name': 'Tflx_conv',   'shortname': 'Tflx_conv',   'units': '-', 'cumulative': True},
    'p212031': {'variable_name': 'dqdt_conv',   'shortname': 'dqdt_conv',   'units': '-', 'cumulative': True},
    'p212032': {'variable_name': 'qflx_conv',   'shortname': 'qflx_conv',   'units': '-', 'cumulative': True},
    'p212033': {'variable_name': 'dco2dt_conv', 'shortname': 'dco2dt_conv', 'units': '-', 'cumulative': True},
    'p212034': {'variable_name': 'co2flx_conv', 'shortname': 'co2flx_conv', 'units': '-', 'cumulative': True},
    'p212035': {'variable_name': 'dch4dt_conv', 'shortname': 'dch4dt_conv', 'units': '-', 'cumulative': True},
    'p212036': {'variable_name': 'ch4flx_conv', 'shortname': 'ch4flx_conv', 'units': '-', 'cumulative': True},

    # cloud
    'p212037': {'variable_name': 'dTdt_cloud',  'shortname': 'dTdt_cloud',  'units': '-', 'cumulative': True},
    'p212038': {'variable_name': 'Tflx_cloud',  'shortname': 'Tflx_cloud',  'units': '-', 'cumulative': True},
    'p212039': {'variable_name': 'dqdt_cloud',  'shortname': 'dqdt_cloud',  'units': '-', 'cumulative': True},
    'p212040': {'variable_name': 'qflx_cloud',  'shortname': 'qflx_cloud',  'units': '-', 'cumulative': True},
}

#Surface level
dict_surface = {
    "168": {"variable_name": "2 metre dewpoint temperature", "shortname": "d2m", "units": "K", "cumulative": False},
    "167": {"variable_name": "2 metre temperature", "shortname": "t2m", "units": "K", "cumulative": False},
    "165": {"variable_name": "10 metre U wind component", "shortname": "u10", "units": "m s-1", "cumulative": False},
    "166": {"variable_name": "10 metre V wind component", "shortname": "v10", "units": "m s-1", "cumulative": False},
    "171007": {"variable_name": "100 metre U wind component", "shortname": "u100", "units": "m s-1", "cumulative": False},
    "228246": {"variable_name": "100 metre V wind component", "shortname": "v100", "units": "m s-1", "cumulative": False},
    "228082": {"variable_name": "Accumulated Carbon Dioxide Ecosystem Respiration", "shortname": "aco2rec", "units": "kg m-2", "cumulative": True},
    "228081": {"variable_name": "Accumulated Carbon Dioxide Gross Primary Production", "shortname": "aco2gpp", "units": "kg m-2", "cumulative": True},
    "228080": {"variable_name": "Accumulated Carbon Dioxide Net Ecosystem Exchange", "shortname": "aco2nee", "units": "kg m-2", "cumulative": True},
    "174": {"variable_name": "Albedo (climatological)", "shortname": "al", "units": "(0 – 1)", "cumulative": False},
    "159": {"variable_name": "Boundary layer height", "shortname": "blh", "units": "m", "cumulative": False},
    "210064": {"variable_name": "CO2 column-mean molar fraction", "shortname": "tcco2", "units": "ppm", "cumulative": False},
    "210061": {"variable_name": "Carbon dioxide mass mixing ratio", "shortname": "co2", "units": "kg kg-1", "cumulative": False},
    "228023": {"variable_name": "Cloud base height", "shortname": "cbh", "units": "m", "cumulative": False},
    "143": {"variable_name": "Convective precipitation", "shortname": "cp", "units": "m", "cumulative": False},
    "182": {"variable_name": "Evaporation", "shortname": "e", "units": "m of water equivalent", "cumulative": False},
    "243": {"variable_name": "Forecast albedo", "shortname": "fal", "units": "(0 – 1)", "cumulative": False},
    "245": {"variable_name": "Forecast logarithm of surface roughness for heat", "shortname": "flsr", "units": "Numeric", "cumulative": False},
    "228078": {"variable_name": "GPP coefficient from Biogenic Flux Adjustment System", "shortname": "gppbfas", "units": "-", "cumulative": False},
    "129": {"variable_name": "Geopotential", "shortname": "z", "units": "m", "cumulative": False},
    "228046": {"variable_name": "Height of convective cloud top", "shortname": "hcct", "units": "m", "cumulative": False},
    "188": {"variable_name": "High cloud cover", "shortname": "hcc", "units": "(0 – 1)", "cumulative": False},
    "172": {"variable_name": "Land-sea mask", "shortname": "lsm", "units": "(0 – 1)", "cumulative": False},
    "142": {"variable_name": "Large-scale precipitation", "shortname": "lsp", "units": "m", "cumulative": False},
    "66": {"variable_name": "Leaf area index, low vegetation", "shortname": "lai_lv", "units": "m2 m-2", "cumulative": False},
    "67": {"variable_name": "Leaf area index, high vegetation", "shortname": "lai_hv", "units": "m2 m-2", "cumulative": False},
    "234": {"variable_name": "Logarithm of surface roughness length for heat (climatological)", "shortname": "lsrh", "units": "Numeric", "cumulative": False},
    "186": {"variable_name": "Low cloud cover", "shortname": "lcc", "units": "(0 – 1)", "cumulative": False},
    "151": {"variable_name": "Mean sea level pressure", "shortname": "msl", "units": "Pa", "cumulative": False},
    "187": {"variable_name": "Medium cloud cover", "shortname": "mcc", "units": "(0 – 1)", "cumulative": False},
    "228235": {"variable_name": "Most-unstable CAPE", "shortname": "mucape", "units": "J kg-1", "cumulative": False},
    "228236": {"variable_name": "Most-unstable CIN", "shortname": "mucin", "units": "J kg-1", "cumulative": False},
    "235": {"variable_name": "Skin temperature", "shortname": "skt", "units": "K", "cumulative": False},
    "139": {"variable_name": "Soil temperature level 1", "shortname": "stl1", "units": "K", "cumulative": False},
    "170": {"variable_name": "Soil temperature level 2", "shortname": "stl2", "units": "K", "cumulative": False},
    "183": {"variable_name": "Soil temperature level 3", "shortname": "stl3", "units": "K", "cumulative": False},
    "236": {"variable_name": "Soil temperature level 4", "shortname": "stl4", "units": "K", "cumulative": False},
    "147": {"variable_name": "Surface latent heat flux", "shortname": "slhf", "units": "J m-2", "cumulative": True},
    "175": {"variable_name": "Surface long-wave (thermal) radiation downwards", "shortname": "strd", "units": "J m-2", "cumulative": True},
    "177": {"variable_name": "Surface net long-wave (thermal) radiation", "shortname": "str", "units": "J m-2", "cumulative": True},
    "176": {"variable_name": "Surface net short-wave (solar) radiation", "shortname": "ssr", "units": "J m-2", "cumulative": True},
    "134": {"variable_name": "Surface pressure", "shortname": "sp", "units": "Pa", "cumulative": False},
    "173": {"variable_name": "Surface roughness (climatological)", "shortname": "sr", "units": "m", "cumulative": False},
    "146": {"variable_name": "Surface sensible heat flux", "shortname": "sshf", "units": "J m-2", "cumulative": True},
    "169": {"variable_name": "Surface short-wave (solar) radiation downwards", "shortname": "ssrd", "units": "J m-2", "cumulative": True},
    "130": {"variable_name": "Temperature", "shortname": "t", "units": "K", "cumulative": False},
    "164": {"variable_name": "Total cloud cover", "shortname": "tcc", "units": "(0 – 1)", "cumulative": False},
    "78": {"variable_name": "Total column cloud liquid water", "shortname": "tclw", "units": "kg m-2", "cumulative": False},
    "228089": {"variable_name": "Total column rain water", "shortname": "tcrw", "units": "kg m-2", "cumulative": False},
    "228090": {"variable_name": "Total column snow water", "shortname": "tcsw", "units": "kg m-2", "cumulative": False},
    "137": {"variable_name": "Total column vertically-integrated water vapour", "shortname": "tcwv", "units": "kg m-2", "cumulative": False},
    "136": {"variable_name": "Total column water", "shortname": "tcw", "units": "kg m-2", "cumulative": False},
    "228": {"variable_name": "Total precipitation", "shortname": "tp", "units": "m", "cumulative": False},
    "213": {"variable_name": "Vertically integrated moisture divergence", "shortname": "vimd", "units": "kg m-2", "cumulative": False},
    "39": {"variable_name": "Volumetric soil water layer 1", "shortname": "swvl1", "units": "m3 m-3", "cumulative": False},
    "40": {"variable_name": "Volumetric soil water layer 2", "shortname": "swvl2", "units": "m3 m-3", "cumulative": False},
    "41": {"variable_name": "Volumetric soil water layer 3", "shortname": "swvl3", "units": "m3 m-3", "cumulative": False},
    "42": {"variable_name": "Volumetric soil water layer 4", "shortname": "swvl4", "units": "m3 m-3", "cumulative": False},
}

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       F U N C T I O N S 
#------------------------------------------------------------------------------------------------------------------------------------------------------

def process_sfc(rootdir, EXPVER, variables,dict_surface,after_suffix=None):

    '''
    This function processes the surface dataset by applying necessary transformations and calculations.

    rootdir: The root directory of the dataset.
    EXPVER: The experiment version.
    variables: Variables to process, if None, compute all
    dict_surface: settings on sfc-variables
    after_suffix: If not None, add this suffix to the end of the output file name. Standard option = None.
    '''
    
    #Load data
    if "_LS" in EXPVER:
        
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    date_pattern = re.compile(r"_sfc_ll_(\d{8})\.nc$")  
    files = glob.glob(f"{rootdir}/{EXPVER_c}/SFC/{EXPVER}_sfc_ll_*.nc")
    files_with_date = [f for f in files if date_pattern.search(os.path.basename(f))]
    
    for file in files_with_date: 
        
        date = date_pattern.search(os.path.basename(file)).group(1)
        ds = xr.open_mfdataset([file,f'{rootdir}/{EXPVER_c}/SFC/{EXPVER}_sfc_ll_co2vars_{date}.nc',f'{rootdir}/{EXPVER_c}/SFC/{EXPVER}_sfc_ll_gpp_{date}.nc'], chunks={'longitude': -1, 'latitude': -1, 'time': 50})

        print(f'Processing file for date {date}')
        
        #Output dataset
        ds_output = xr.Dataset(coords={'time': ds.time.values})
        
        #Loop over variables in dataset and compare to settings in the dictionary
        if variables is None:
            data_vars = list(ds.data_vars)
        else:
            data_vars = variables
            
        id_list = dict_surface.keys()
        shortname_list = [dict_surface[str(id)]['shortname'] for id in id_list]
    
        for id in data_vars: 
            
            print(f'Processing {id}')
            
            #Check occurrence of variable and settings
            if id in shortname_list:
                key_value = dict_surface[next(key for key in id_list if dict_surface[key]['shortname'] == id)]
            else: 
                raise('Variable does not occur in current list, check files')

            #Load options
            var_name = key_value['variable_name']
            short_name = key_value['shortname']
            units = key_value['units']
            cumulative = key_value['cumulative']
            
            #Uncumulative if necessary
            if cumulative == True:
                print(f'Variable {id} is cumulative, applying deaccumulation')
                
                var_sel = ds[id]
            
                var_diff = var_sel.diff(dim='time',label='upper')
                
                var_deaccumulated = var_diff / 3600
                
                # var_deaccumulated = var_deaccumulated.isel(time=slice(1, None)) #!!

                ds_output[id] = var_deaccumulated
                
                if units == 'J m-2':
                    ds_output[id].attrs['units'] = 'W/m2'
                else:
                    ds_output[id].attrs['units'] = units + '_uncumulated'
            
            else:
                ds_output[id] = ds[id]
   
            #------------------------------------------------
            
            #Conversions
            if id == 'co2':

                factor = 658139.7198332178 # (factor=1e6*mmair/mmco2)

                ds_output[id] = ds_output[id] * factor
                ds_output[id].attrs['units'] = 'ppm'

            elif id == "aco2nee":

                ds_output[id] = ds_output[id]*-22.72e6
                ds_output[id].attrs['units'] = 'mumol/m2/s'

            elif id == 'aco2rec': 
                ds_output[id] = ds_output[id]*-22.72e6
                ds_output[id].attrs['units'] = 'mumol/m2/s'

            elif id == 'aco2gpp':

                #Check if gppbfas is available
                if 'gppbfas' not in ds_output.data_vars:
                    raise ValueError(f"Variable gppbfas is required for correction of aco2gpp but is not available in the dataset.")
                else:
                    print('Applying correction to aco2gpp using gppbfas')
                    #Load reference (need to copy the correction from t_0 to all)
                    factor = ds_output['gppbfas']
                                
                    #Loop through dates and place correction factor for each date for all instances
                    f = factor.groupby('time.date').map(lambda g: g.isel(time=0).broadcast_like(g))

                    #Correct
                    ds_output[id] = ds_output[id] * f *- 22.72e6
                    ds_output[id].attrs['units'] = 'mumol/m2/s'

        #------------------------------------------------

        #Calculations (Radiation components, and EF)

        #Radiation Components

        #Qnet
        if 'ssr' in data_vars and 'str' in data_vars:
            ds_output['qnet'] = ds_output['ssr'] + ds_output['str']
            ds_output['qnet'].attrs['units'] = 'W/m2'
            ds_output['qnet'].attrs['standard_name'] = 'surface_net_radiation'
            print(f'Calculated qnet')

        #SWout
        if 'ssr' in data_vars and 'ssrd' in data_vars:
            ds_output['ssru'] = ds_output['ssr'] - ds_output['ssrd']
            ds_output['ssru'].attrs['units'] = 'W/m2'
            ds_output['ssru'].attrs['standard_name'] = 'surface_shortwave_radiation_upward'
            print(f'Calculated ssru')

        #LWout
        if 'str' in data_vars and 'strd' in data_vars:
            ds_output['stru'] = ds_output['str'] - ds_output['strd']
            ds_output['stru'].attrs['units'] = 'W/m2'
            ds_output['stru'].attrs['standard_name'] = 'surface_longwave_radiation_upward'
            print(f'Calculated stru')

        #EF
        if 'slhf' in data_vars and 'sshf' in data_vars:
            ds_output['EF'] = -1*ds_output['slhf']/(-1*ds_output['slhf'] + -1*ds_output['sshf'])
            ds_output['EF'].attrs['units'] = '(0 - 1)'
            ds_output['EF'].attrs['standard_name'] = 'evaporation_fraction'
            print(f'Calculated EF')

        #e_sat
        if 't2m' in data_vars:
            ds_output['esat'] = 0.611e3 * np.exp(17.2694 * (ds_output['t2m'] - 273.16) / (ds_output['t2m'] - 35.86))
            ds_output['esat'].attrs['units'] = 'Pa'
            ds_output['esat'].attrs['standard_name'] = 'saturation_vapor_pressure'
            print(f'Calculated esat')
        
        #e
        if 'esat' in ds_output.data_vars and 't2m' in data_vars and 'd2m' in data_vars and 'sp' in data_vars:
            
            #e
            ds_output['e'] = 0.611e3 * np.exp(17.2694 * (ds_output['d2m'] - 273.16) / (ds_output['d2m'] - 35.86))
            ds_output['e'].attrs['units'] = 'Pa'
            ds_output['e'].attrs['standard_name'] = 'vapor_pressure'
            print(f'Calculated e')
        
        #q
        if 'e' in ds_output.data_vars and 'sp' in data_vars and 't2m' in data_vars:
            
            #w = epsilon * e / (p - e)
            epsilon = 0.622
            
            ds_output['q'] = epsilon * ds_output['e'] / (ds_output['sp'] - ds_output['e'])
            ds_output['q'].attrs['units'] = 'kg/kg'
            ds_output['q'].attrs['standard_name'] = 'specific_humidity'
            print(f'Calculated q')
        
        #VPD
        if 'sp' in data_vars and 'esat' in ds_output.data_vars and 'e' in ds_output.data_vars:
            
            ds_output['VPD'] = ds_output['esat'] - ds_output['e']
            ds_output['VPD'].attrs['units'] = 'Pa'
            ds_output['VPD'].attrs['standard_name'] = 'vapor_pressure_deficit'
            print(f'Calculated VPD')
        
        #Create outdir
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/SFC/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/SFC/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/SFC/Processed"

        #Save
        ds_output = ds_output.astype("float32")
        with ProgressBar():
            if after_suffix == None:
                ds_output.to_netcdf(f"{outdir}/{EXPVER}_sfc_complete_{date}.nc",engine='h5netcdf')
            else:
                ds_output.to_netcdf(f"{outdir}/{EXPVER}_sfc_{after_suffix}_{date}.nc",engine='h5netcdf')

        #Close
        ds.close()
        ds_output.close()
            
        print(f"Saved to {outdir}")

#------------------------------------------------------------------------------------------------------------------------------------------------------

def process_ml(rootdir, EXPVER, model_def, variables, dict_model,tendency_in_data=True,after_suffix=None):
    ''' 
    This function processes the model level dataset by applying necessary transformations and calculations.

    rootdir: The root directory of the dataset.
    EXPVER: The experiment version.
    model_def: The model definition containing the L137 level model definition of IFS, including a and b constants. (.csv file)
    variables: variables to process, if None compute all
    dict_model: settings model parameters
    height_interpolation: Boolean flag indicating whether to perform height interpolation. Standard option = True. 
    tendency_in_data: Boolean flag indicating whether tendency data is included. Standard option = True.
    after_suffix: If not None, add this suffix to the end of the output file name. Standard option = None.
    '''

    #Load data
    if "_LS" in EXPVER:

        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    date_pattern = re.compile(r"_ml_ll_(\d{8})\.nc$")  
    files = glob.glob(f"{rootdir}/{EXPVER_c}/ML/{EXPVER}_ml_ll_*.nc")
    files_with_date = [f for f in files if date_pattern.search(os.path.basename(f))]

    for file in files_with_date: 
        
        chunks = {"time": 3} # ,"level": 1, "latitude": 96, "longitude": 96}
        
        date = date_pattern.search(os.path.basename(file)).group(1)
        ds = xr.open_dataset(file,chunks=chunks)
        time = ds.time.values

        print(f'Processing file for date {date}')
        
        #Output dataset
        ds_output = xr.Dataset(coords={'time':time})
        
        #Update longitude
        ds = ds.assign_coords(longitude= ds.longitude - 360)
        
        #Loop over variables in dataset and compare to settings in the dictionary
        if variables is None:
            data_vars = list(ds.data_vars)
        else:
            data_vars = list(variables)   # or variables.copy()

        
        id_list =  dict_model.keys()
        shortname_list = [dict_model[str(id)]['shortname'] for id in id_list]
        
        for id in data_vars: 
            
            print(f'Processing {id}')

            #------ R E N A M E ------ 
            
            #Check occurrence of variable and settings
            if id in shortname_list:
                key_value = dict_model[next(key for key in id_list if dict_model[key]['shortname'] == id)]
            elif id in id_list: 
                key_value = dict_model[id]
                
                print(f'Renaming {id} to {key_value["shortname"]}')
                
                #Rename
                ds = ds.rename({id: key_value['shortname']})
                ds.attrs['long_name'] = key_value['variable_name']
                ds.attrs['units'] = key_value['units']
                id = key_value['shortname']

            else:
                raise ValueError(f"Variable {id!r} does not occur in current list, check files")

            #------ U N C U M U L A T E ------ 
            
            #Load options
            var_name = key_value['variable_name']
            short_name = key_value['shortname']
            units = key_value['units']
            cumulative = key_value['cumulative']
            
            #Uncumulative if necessary
            if cumulative == True:
                print(f'Variable {id} is cumulative, applying deaccumulation')

                var_sel = ds[id]
            
                var_diff = var_sel.diff(dim='time',label='upper')
                
                var_deaccumulated = var_diff / 3600
                
                #var_deaccumulated = var_deaccumulated.isel(time=slice(1, None)) #!!

                ds_output[id] = var_deaccumulated

            else:
                ds_output[id] = ds[id]
            
            #------ C O N V E R S I O N S ------    
            
            #Flux conversion after density is calculated (see below)  

            #Conversions
            if id == 'co2':

                factor = 658139.7198332178 # (factor=1e6*mmair/mmco2)

                ds_output[id] = ds_output[id] * factor
                ds_output[id].attrs['units'] = 'ppm'

            elif id == 'ch4':

                factor = 1805760598.5037408 # (factor=1e9*mmair/mmch4)

                ds_output[id] = ds_output[id] * factor
                ds_output[id].attrs['units'] = 'ppb'
            
            elif id in ['dco2dt_dyn', 'dco2dt_diff', 'dco2dt_conv']:

                factor = 658139.7198332178 # (factor=1e6*mmair/mmco2)

                ds_output[id] = ds_output[id] * factor * 3600 # Convert from per second to per hour
                ds_output[id].attrs['units'] = 'ppm/h'
            
            elif id == 'q':
                
                factor = 1000 # Convert from kg/kg to g/kg

                ds_output[id] = ds_output[id] * factor
                ds_output[id].attrs['units'] = 'g/kg'

            elif id in ['umfa','dmfa']:

                ds_output[id] = ds_output[id] * 3600 # Convert from per second to per hour
                ds_output[id].attrs['units'] = 'm/h'

        #------------------------------------------------
        if {'t', 'q'}.issubset(data_vars):
            
            #Open LNSP
            ds_lnsp = xr.open_dataset(
                f'{rootdir}/{EXPVER_c}/ML/{EXPVER}_ml_ll_lnsp_{date}.nc',chunks=chunks)
            
            print('Calculating pressure and height arrays')

            lnsp = ds_lnsp['lnsp']  # (time, lat, lon)

            # ------------------------------------------------------------------
            # A/B half-level coefficients
            # ------------------------------------------------------------------
            print('Loading model definition for a/b coefficients')

            df = pd.read_csv(model_def).set_index('n').sort_index()

            levels = ds_output['level'].values
            nlev = levels.size
            top_lev = min(ds.level.values)

            #Half-level indices: need nlev+1 half-levels to get nlev full levels
            a = xr.DataArray(df['a'].to_numpy(), dims='level')
            b = xr.DataArray(df['b'].to_numpy(), dims='level')

            #------------------------------------------------------------------------------------
            #                                           PRESSURE
            #------------------------------------------------------------------------------------
            print('Calculating pressure levels')

            # Surface pressure — keep it lazy, no expand_dims broadcast
            sp = np.exp(lnsp)  # (time, lat, lon)
            
            # Calculate half-level pressures
            p_half = a + b * sp

            # Calculate full-level pressures by averaging adjacent half-levels
            p_full = (p_half.isel(level=slice(0, -1)) + p_half.isel(level=slice(1, None))) / 2
            p_full = p_full.assign_coords(level=np.arange(1, 138))
                    
            # ------------------------------------------------------------------
            # Assign back cleanly
            # ------------------------------------------------------------------
            
            #Flip level and time dimensions from (level, time, lat, lon) to (time, level, lat, lon)
            p_full = p_full.transpose("time", "level", "latitude", "longitude")
            
            ds_output['p'] = xr.DataArray(
                p_full.sel(level=ds_output.level),  # Ensure levels match
                dims=('time', 'level',"latitude", "longitude"),
                coords={
                    "time":      ds_output.time,
                    "level":     ds_output.level,
                    "latitude": ds_output.latitude,
                    "longitude": ds_output.longitude,
                },
            )
            ds_output['p'].attrs.update(units='Pa', standard_name='pressure')

            data_vars.append('p')
            print('Calculated p')
            

            #------------------------------------------------------------------------------------
            #                                            HEIGHT
            #------------------------------------------------------------------------------------

            #Hydrostatic balance
            p_sel = ds_output['p']                                          # (time, level, lat, lon)
            dp   =  p_full.diff(dim="level").sel(level=ds_output.level)     # (time, level, lat, lon)
            t    = ds_output['t']                                           # (time, level, lat, lon)
            q    = ds_output['q']                                           # (time, level, lat, lon)
                    
            #Constants
            rd, g = 287.06, 9.8066
            
            dz = (dp.drop_vars("longitude") * rd * t * (1 + 0.61 * q / 1000)) / (p_sel.drop_vars("longitude") * g)
            dz = dz.assign_coords(longitude=ds_output.longitude)
            
            #surface level
            h_0 = 0
            dz_reversed = dz.sortby("level", ascending=False)     # bottom -> top
            z_accumulated = dz_reversed.cumsum('level') 
            z_accumulated = z_accumulated.sortby('level')
            h_levels = h_0 + z_accumulated
            
            ds_output['z'] = h_levels
            ds_output['z'].attrs.update(units='m', standard_name='height')

            data_vars.append('z')
            print('Calculated z')
            

        #------------------------------------------------
        
        #Calculate density and convert omega
        if {'w', 'p'}.issubset(data_vars):
            
            #Convert omega to vertical velocity in m/s
            rd, g = 287.06, 9.81

            #rho
            p_sel = ds_output['p']                                          # (time, level, lat, lon)
            t    = ds_output['t']                                           # (time, level, lat, lon)
            q    = ds_output['q']                                           # (time, level, lat, lon)
            
            rho = p_sel / (rd * t * (1 + 0.61 * q / 1000))
            
            ds_output['density'] = rho
            ds_output['density'].attrs.update(units='kg/m3', standard_name='air_density')
            
            data_vars.append('density')
            print('Calculated density')
            
            #Convert omega
            ds_output['w'] = ds_output['w'] / -(rho * g)
            ds_output['w'].attrs.update(units='m/s', standard_name='vertical_velocity')
            print('Converted omega to w')
            
        
        #------------------------------------------------

        #Potential temperature
        if {'t', 'p'}.issubset(data_vars):

            #Potential Temperature
            cp = 1005.7
            Rd = 287.05
            p0 = 100000
            kappa = Rd/cp

            #Create dataset
            theta = ds_output.t * (p0/ds_output.p)**kappa

            ds_output['theta'] = theta
            ds_output['theta'].attrs['units'] = 'K'
            ds_output['theta'].attrs['standard_name'] = 'potential_temperature'

            print(f'Calculated theta')
            data_vars.append('theta')
                
        #------------------------------------------------

        #Wind speed
        if {'u', 'v'}.issubset(data_vars):
            ws = np.sqrt(ds_output.u**2 + ds_output.v**2)
            
            ds_output['ws'] = ws
            ds_output['ws'].attrs['units'] = 'm/s'
            ds_output['ws'].attrs['standard_name'] = 'wind_speed'

            print(f'Calculated ws')
            data_vars.append('ws')
            

        #------------------------------------------------
        
        #Wind direction
        if {'u', 'v'}.issubset(data_vars):
            WD = (np.arctan2(-ds_output.u,-ds_output.v) * 180 / np.pi) % 360

            ds_output['wdir'] = WD
            ds_output['wdir'].attrs['units'] = 'degrees'
            ds_output['wdir'].attrs['standard_name'] = 'wind_direction'

            print(f'Calculated wdir')
            data_vars.append('wdir')
            

        #------------------------------------------------
        
        #Correct CO2 fluxes to ppm/h if in dataset
        if tendency_in_data:
            
            print('Converting CO2 fluxes to ppm/h')

            factor_1 = (ds_output.density)
            factor_2 = 658139.7198332178*-1 # (factor=1e6*mmair/mmco2)
            
            # flux_convective = (flux_convective/(1000/density_calc))*(28.9644/44.01)*1e9*-1 #ppm m h-1
            ds_output['co2flx_dyn'] = ds_output['co2flx_dyn']/factor_1
            ds_output['co2flx_dyn'] = ds_output['co2flx_dyn']*factor_2 * 3600
            ds_output['co2flx_dyn'].attrs['units'] = 'ppm m h-1'

            ds_output['co2flx_diff'] = ds_output['co2flx_diff']/factor_1
            ds_output['co2flx_diff'] = ds_output['co2flx_diff']*factor_2 * 3600
            ds_output['co2flx_diff'].attrs['units'] = 'ppm m h-1'

            ds_output['co2flx_conv'] = ds_output['co2flx_conv']/factor_1
            ds_output['co2flx_conv'] = ds_output['co2flx_conv']*factor_2 * 3600
            ds_output['co2flx_conv'].attrs['units'] = 'ppm m h-1'

            print('Converted CO2 fluxes to ppm/h')    
        
        #-------------------------------

        #Export
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/ML/Processed"):
            os.makedirs(f"{rootdir}/{EXPVER_c}/ML/Processed")
        
        outdir = f"{rootdir}/{EXPVER_c}/ML/Processed"
        ds_output = ds_output.astype("float32")

        with ProgressBar():         
            if after_suffix == None:
                ds_output.to_netcdf(f'{outdir}/{EXPVER_c}_ml_complete_{date}.nc',engine='h5netcdf')
            else:     
                ds_output.to_netcdf(
                f'{outdir}/{EXPVER_c}_ml_{after_suffix}_{date}.nc',engine='h5netcdf')
                               
        print(f'Exported processed file for {date}')
    

#------------------------------------------------------------------------------------------------------------------------------------------------------

def aggregate_on_time(rootdir,EXPVER,datatype,reference_dates,delta_time_hr,file_suffix,after_suffix,export=False):

    """
    This function aggregates the data to the reference dates.

    rootdir: str : root directory
    EXPVER: str : experiment identifier (e.g., ia61)
    datatype: str : 'sfc' or 'ml'
    reference_dates: array : reference dates
    file_suffix: str : suffix of the file to denote the type of file (e.g., complete)
    after_suffix: str : suffix of the file to denote the type of file after aggregation (e.g., aggregated)

    """

    #Load data
    if "_LS" in EXPVER:
    
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
        
    ds = xr.open_dataset(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed/{EXPVER}_{datatype}_{file_suffix}.nc")

    #Transpose to local time (e.g., 4 hours from UTC to LT for the Amazon)
    ds['time'] = pd.to_datetime(ds['time']) + timedelta(hours=delta_time_hr) #.values) 
    #Mean and std
    sub = ds.sel(time=ds['time'].dt.strftime('%Y-%m-%d').isin(reference_dates)) 
    sub_mean = sub.groupby('time.hour').mean(dim='time')
    sub_std = sub.groupby('time.hour').std(dim='time')

    #Rename
    sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
    sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})

    #Combine datasets into one
    ds_aggregated = xr.merge([sub_mean, sub_std])
    ds_aggregated = ds_aggregated[sorted(ds_aggregated.data_vars)]

    #Create outdir
    if export == True:
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"

        #Save
        with ProgressBar():
            ds_aggregated.to_netcdf(f"{outdir}/{EXPVER}_{datatype}_{after_suffix}.nc",engine='h5netcdf')

        #Close
        ds.close()
        ds_aggregated.close()
        sub.close()
        sub_mean.close()
        sub_std.close()

        print(f" Aggregated and saved in {outdir}")
    else:
        return ds_aggregated


#------------------------------------------------------------------------------------------------------------------------------------------------------

def interpolate_to_heights(rootdir,EXPVER,variables,pre_suffix,after_suffix,z_ref,export=False):
    """
    Interpolate a field defined on model full levels (via their geometric heights)
    onto requested geometric height levels (in meters).

    Args:
        rootdir : str : root directory
        EXPVER : str : experiment version
        variables : list of variable names to interpolate
        pre_suffix: suffix of the file
        after_suffix : suffix for output files
        z_ref : 1D array of target height levels [m] (ascending)
        export: save the outputted file. Default is false

    Returns:
        var_hl : array with level axis replaced by len(z_ref)
    """

    #Load data
    if "_LS" in EXPVER:
        
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    #Wchunks = {"time": 1, "level": -1}
    ds = xr.open_dataset(
            f"{rootdir}/{EXPVER_c}/ML/Processed/{EXPVER}_ml_{pre_suffix}.nc") #,chunks=chunks)
    z = ds['z']
    
    #Output dataset
    ds_z = xr.Dataset(coords={
        "time": ds.time,
        "latitude": ds.latitude,
        "longitude": ds.longitude,
        "z": z_ref,
    })
    
    ds_z['z'].attrs.update({'units': 'm', 'long_name': 'Height from the surface'})
            
    #Interpolation     
    
    flux_vars = ['co2flx_conv', 'co2flx_diff', 'co2flx_dyn']
       
    for var in variables: 
        
        if var not in flux_vars:
            
            #print(f'Interpolating {var}')
            
            da = ds[var].chunk({"level": -1})
            
            ds_z[var] = xr.apply_ufunc(                         #Apply a custom function
                lambda y_lev, z_lev: interp1d(                  #For every profile, build y(z)
                    z_lev,                                      #x-values: heights at model lev
                    y_lev,                                      #y-values: variable at model lev
                    kind="linear",                              #Linear interpolation
                    bounds_error=False,                         #Don't raise error when extrapolating=
                    fill_value="extrapolate",                   #Extrapolate when outside of bounds
                    assume_sorted=False,                        #Sort z_lev when needed
                )(z_ref),                                       #Evaluate the interpolation function at the reference heights
                da,                                             #Input array: variable at model levels (y_lev) y-values
                z,                                              #Input array: heights at model levels (z_lev) x-values
                input_core_dims=[["level"], ["level"]],         #Dimensions of the input arrays
                output_core_dims=[["z"]],                       #Dimensions of the output array
                vectorize=True,                                 #Apply the function to every profile
                dask="parallelized",                            #Use dask for parallelization
                output_dtypes=[da.dtype],                       #Output dtype is the same as input dtype
                dask_gufunc_kwargs={
                "output_sizes": {"z": len(z_ref)}
    }                                                           #Output size along the new dimension z is the length of z_ref
            ).assign_coords(z=z_ref)                            #Assign the reference heights as coordinates for the output
        
            #In essence, the above does the following loops:
            # for it in ds.time:
            #     for ilat in ds.latitude:
            #         for ilon in ds.longitude:
            #             ifs_it = ds.sel(time=it, latitude=ilat, longitude=ilon)[var]
            #             ifs_it = ifs_it.assign_coords({
            #                 "z": ("level", z.sel(time=it, latitude=ilat, longitude=ilon).values)
            #             }).swap_dims({"level": "z"})
                                                
            #             #Interp
            #             ifs_it = ifs_it.interp(z=z_ref, kwargs={"fill_value": "extrapolate"})
            #             ds_out.loc[dict(time=it, latitude=ilat, longitude=ilon)] = ifs_it
            
            # ds_z[var] = ds_out[var]
            
        else:
            
            #Fluxes are on half-levels, so we need to calculate the values at half-levels first by averaging adjacent full levels, and then interpolate to the reference heights
            da = 0.5 * (
            ds[var].isel(level=slice(None, -1)) +
            ds[var].isel(level=slice(1, None))
            )

            z_mid = 0.5 * (
                z.isel(level=slice(None, -1)) +
                z.isel(level=slice(1, None))
            )

            da = da.chunk({"level": -1})
            z_mid = z_mid.chunk({"level": -1})

            ds_z[var] = xr.apply_ufunc(
                lambda y_lev, z_lev: interp1d(
                    z_lev,
                    y_lev,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=False,
                )(z_ref),
                da,
                z_mid,
                input_core_dims=[["level"], ["level"]],
                output_core_dims=[["z"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[da.dtype],
                dask_gufunc_kwargs={"output_sizes": {"z": len(z_ref)}},
            ).assign_coords(z=z_ref)
               
    #Create outdir
    if not os.path.exists(f"{rootdir}/{EXPVER_c}/ML/Processed"):
        os.makedirs(f"{rootdir}/{EXPVER_c}/ML/Processed")

    outdir = f"{rootdir}/{EXPVER_c}/ML/Processed"
    ds_z = ds_z.astype("float32")

    #Save
    if export == True:
        
        with ProgressBar():
            ds_z.to_netcdf(f"{outdir}/{EXPVER}_ml_{after_suffix}.nc",engine='h5netcdf')

        print(f"Interpolated to height and saved in {outdir}")

        ds_z.close()

    else:
        return ds_z


#------------------------------------------------------------------------------------------------------------------------------------------------------


def interpolate_IFS_data_on_time(ds_input,frequency):

    '''
    Interpolates IFS data onto a uniform time grid using PCHIP interpolation.
    
    Parameters:
    ds_input (xarray.Dataset): Input dataset with a 'time' dimension.
    frequency (str): Frequency string for the new uniform time grid (e.g., '15T' for 15 minutes).
    
    Returns:
    xarray.Dataset: Dataset interpolated onto the new uniform time grid.
    
    '''
    
    def pchip_1d(y, x, x_new):
        p = PchipInterpolator(x, y, extrapolate=False)
        return p(x_new)

    out_per_day = []

    # group by day (each group keeps all other dims!)
    for day, da_day in ds_input.groupby("time.date"):
        
        da_day = da_day.chunk({"time": -1})

        # new uniform time grid for that day
        new_time = pd.date_range(
            start=pd.Timestamp(da_day.time.min().item()),
            end=pd.Timestamp(da_day.time.max().item()),
            freq=frequency
        )

        # numeric seconds for PCHIP
        x = da_day.time #.values.astype("datetime64[ns]").astype("int64") / 1e9
        x_new = new_time #.values.astype("datetime64[ns]").astype("int64") / 1e9

        # apply along time, vectorize over all other dims
        da_new = xr.apply_ufunc(
            pchip_1d,
            da_day,
            xr.DataArray(x, dims=["time"]),
            xr.DataArray(x_new, dims=["time_new"]),
            input_core_dims=[["time"], ["time"], ["time_new"]],
            output_core_dims=[["time_new"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da_day.dtype],
        )

        # replace time_new with a real datetime coordinate named "time"
        da_new = da_new.rename(time_new="time").assign_coords(time=new_time)

        out_per_day.append(da_new)

    # concatenate all days back together
    ds_interp = xr.concat(out_per_day, dim="time").sortby("time")

    return ds_interp

#------------------------------------------------------------------------------------------------------------------------------------------------------

def subsetter_km(rootdir,EXPVER,ds_path,datatype,x,y,size_km,after_suffix,export=False): 
    
    '''
    Subset the dataset around a specific point (x, y) with a given step size.

    rootdir: str : root directory
    EXPVER: str : experiment version
    ds_path: str : path to the dataset
    datatype: str : type of the dataset (sfc or ml)
    x: float : x-coordinate of the point of interest (longitude)
    y: float : y-coordinate of the point of interest (latitude)
    size_km: float : size of the subset in kilometers (default is 100 km)
    after_suffix: str : suffix for the output file

    '''
    
    #Load data
    if "_LS" in EXPVER:
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    #Load dataset
    ds = xr.open_dataset(ds_path)

    #Find cell nearest to x,y
    latitude = ds.latitude.values
    longitude = ds.longitude.values
    
    centre_lon_index = int(
        np.nanargmin(np.abs(longitude - x))
    )
    centre_lat_index = int(
        np.nanargmin(np.abs(latitude - y))
    )

    centre_lon = float(longitude[centre_lon_index])
    centre_lat = float(latitude[centre_lat_index])
    
    #Calculate limites of region
    half_size_km = size_km /2
    half_lat_deg = half_size_km / 111.32
    
    cos_lat = np.cos(np.deg2rad(centre_lat))
    
    half_lon_deg = half_size_km / (111.32 * cos_lat)
    
    lon_min = centre_lon - half_lon_deg
    lon_max = centre_lon + half_lon_deg
    
    lat_min = centre_lat - half_lat_deg
    lat_max = centre_lat + half_lat_deg
    
    #Select region
    lon_mask = (
        (ds.longitude >= lon_min)
        & (ds.longitude <= lon_max)
    )

    lat_mask = (
        (ds.latitude >= lat_min)
        & (ds.latitude <= lat_max)
    )

    lon_indices = np.flatnonzero(
        np.asarray(lon_mask.values)
    )
    lat_indices = np.flatnonzero(
        np.asarray(lat_mask.values)
    )
    
    available_left = centre_lon_index - lon_indices[0]
    available_right = lon_indices[-1] - centre_lon_index
    half_n_lon = min(available_left, available_right)

    available_below = centre_lat_index - lat_indices[0]
    available_above = lat_indices[-1] - centre_lat_index
    half_n_lat = min(available_below, available_above)

    lon_start = centre_lon_index - half_n_lon
    lon_stop = centre_lon_index + half_n_lon + 1

    lat_start = centre_lat_index - half_n_lat
    lat_stop = centre_lat_index + half_n_lat + 1

    ds_sub = ds.isel(
        longitude=slice(lon_start, lon_stop),
        latitude=slice(lat_start, lat_stop),
    )

    n_lon = ds_sub.sizes["longitude"]
    n_lat = ds_sub.sizes["latitude"]
    
    #Estimate physical dimensions
    selected_lon = np.asarray(ds_sub.longitude.values)
    selected_lat = np.asarray(ds_sub.latitude.values)

    if n_lon > 1:
        lon_centre_span_km = (
            abs(selected_lon[-1] - selected_lon[0])
            * 111.32
            * cos_lat
        )

        local_lon_spacing_km = float(
            np.nanmedian(np.abs(np.diff(selected_lon)))
            * 111.32
            * cos_lat
        )

        approximate_lon_cell_width_km = (
            lon_centre_span_km + local_lon_spacing_km
        )
    else:
        lon_centre_span_km = 0.0
        local_lon_spacing_km = np.nan
        approximate_lon_cell_width_km = np.nan

    if n_lat > 1:
        lat_centre_span_km = (
            abs(selected_lat[-1] - selected_lat[0])
            * 111.32
        )

        local_lat_spacing_km = float(
            np.nanmedian(np.abs(np.diff(selected_lat)))
            * 111.32
        )

        approximate_lat_cell_width_km = (
            lat_centre_span_km + local_lat_spacing_km
        )
    else:
        lat_centre_span_km = 0.0
        local_lat_spacing_km = np.nan
        approximate_lat_cell_width_km = np.nan
    
    ds_sub.attrs["requested_longitude"] = x
    ds_sub.attrs["requested_latitude"] = y

    ds_sub.attrs["central_grid_longitude"] = centre_lon
    ds_sub.attrs["central_grid_latitude"] = centre_lat

    ds_sub.attrs["requested_subset_width_km"] = float(size_km)

    ds_sub.attrs["selected_longitude_cells"] = int(n_lon)
    ds_sub.attrs["selected_latitude_cells"] = int(n_lat)

    ds_sub.attrs["longitude_centre_span_km"] = float(
        lon_centre_span_km
    )
    ds_sub.attrs["latitude_centre_span_km"] = float(
        lat_centre_span_km
    )

    ds_sub.attrs["approximate_longitude_cell_coverage_km"] = float(
        approximate_lon_cell_width_km
    )
    ds_sub.attrs["approximate_latitude_cell_coverage_km"] = float(
        approximate_lat_cell_width_km
    )

    print(
        "Requested location: "
        f"longitude={x:.4f}, "
        f"latitude={y:.4f}"
    )

    print(
        "Central grid cell: "
        f"longitude={centre_lon:.4f}, "
        f"latitude={centre_lat:.4f}"
    )

    print(
        f"Selected grid: {n_lon} × {n_lat} cells"
    )

    print(
        "Distance between outer grid-cell centres: "
        f"{lon_centre_span_km:.1f} km × "
        f"{lat_centre_span_km:.1f} km"
    )

    print(
        "Approximate area covered by the complete cells: "
        f"{approximate_lon_cell_width_km:.1f} km × "
        f"{approximate_lat_cell_width_km:.1f} km"
    )
    

    #Export
    if export == True:
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"

        #Save
        with ProgressBar():
            ds_sub.to_netcdf(f"{outdir}/{EXPVER}_{datatype}_{after_suffix}.nc",engine='h5netcdf')

        print(f"Created subset in {outdir}")

        ds.close()
        ds_sub.close()
        
    else:
        return ds_sub



#------------------------------------------------------------------------------------------------------------------------------------------------------

def subsetter_shapefile(rootdir,EXPVER,ds_path,datatype,path_shapefile,export=False,after_suffix=None):
    
    '''
    Subset the dataset to the shapefile region using a shapefile.
    
    rootdir: str : root directory
    EXPVER: str : experiment version
    ds_path: str : path to the dataset
    datatype: str : type of the dataset (sfc or ml)
    path_shapefile: str : path to the shapefile (netcdf) containing the region mask
    after_suffix: str : suffix for the output file after subsetting    
    
    '''
    
    #Load data
    if "_LS" in EXPVER:
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    ds = xr.open_dataset(ds_path)

    #Load shapefile (netcdf)
    mask = xr.open_dataset(path_shapefile)
    mask['longitude'] = ds['longitude']
    mask['latitude'] = ds['latitude']
    mask_data = mask['Amazon']

    #Cut data to mask
    ds_sub = ds.where(mask_data == 1, drop=True)
    
    #Export
    if export == True:
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"

        #Save
        with ProgressBar():
            ds_sub.to_netcdf(f"{outdir}/{EXPVER}_{datatype}_{after_suffix}.nc",engine='h5netcdf')

        print(f"Created subset in {outdir}")

        ds.close()
        ds_sub.close()
    else:
        return ds_sub

#------------------------------------------------------------------------------------------------------------------------------------------------------

def subsetter_coordinates(rootdir,EXPVER,ds_path,datatype,lon_min,lon_max,lat_min,lat_max,export=False,after_suffix=None):
    
    '''
    Subset the dataset to the specified coordinates.
    
    rootdir: str : root directory
    EXPVER: str : experiment version
    ds_path: str : path to the dataset
    datatype: str : type of the dataset (sfc or ml)
    lon_min: float : minimum longitude
    lon_max: float : maximum longitude
    lat_min: float : minimum latitude
    lat_max: float : maximum latitude
    after_suffix: str : suffix for the output file after subsetting    
    
    '''
    
    #Load data
    if "_LS" in EXPVER:
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    ds = xr.open_dataset(ds_path)

    #Cut data to coordinates
    ds_sub = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
    
    #Export
    if export == True:
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"

        #Save
        with ProgressBar():
            ds_sub.to_netcdf(f"{outdir}/{EXPVER}_{datatype}_{after_suffix}.nc",engine='h5netcdf')

        print(f"Created subset in {outdir}")

        ds.close()
        ds_sub.close()
    else:
        return ds_sub

#------------------------------------------------------------------------------------------------------------------------------------------------------

def region_mask(rootdir,ref_file,name_region,EXPVER,datatype,export=True):
    '''
    Create a region mask for a region using a shapefile.
    
    rootdir: str : root directory
    ref_file: str : path to the reference file (netcdf) containing the coordinates
    name_region: str : name of the region (e.g., 'Amazon')
    EXPVER: str : experiment version
    datatype: str : type of the dataset (sfc or ml)
    export: bool : whether to export the mask to a netcdf file
    
    '''

    #Load data
    if "_LS" in EXPVER:
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER

    #Reference shapefile
    shapefile = "/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/Reference files/Shapefile/hybas_sa_lev02_v1c.shp" #increasing number behind levXX makes smaller selections, level 3 still has Amazon in full shape

    # Retrieve file dimensions
    file = xr.open_dataset(ref_file)

    longitude = np.array(file.longitude)
    latitude = np.array(file.latitude)

    # Read shapefile
    ds = gpd.read_file(shapefile)  # load shape file

    #Visualise shapefile
    ds.plot(edgecolor='black', facecolor='none')
    #Add region ID in the middle of the region
    for idx, row in ds.iterrows():
        centroid = row.geometry.centroid
        plt.text(centroid.x, centroid.y, ds.index[idx], ha='center', va='center', fontsize=8, path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    plt.show()

    #Select region(s) to build mask for
    ds = ds[ds.index.isin([1])]

    #Confirm
    ds.plot(edgecolor='black', facecolor='none')
    plt.show()

    # Generate mask
    mask = regionmask.mask_3D_geopandas(ds, longitude, latitude)

    #plot mask
    longitude_plot = ((longitude + 180) % 360) - 180
    sort_idx = np.argsort(longitude_plot)

    ds.plot(edgecolor='black', facecolor='none')
    plt.contourf(
        longitude_plot[sort_idx],
        latitude,
        mask.isel(region=0).values[:, sort_idx],
        levels=[0.5, 1.5],
        zorder=5,
        alpha=0.8
    )
    plt.show()

    #In mask, set TRUE = 1 and FALSE = 0
    mask_proper = mask.sel(region=1)
    mask_proper = mask_proper.where(mask_proper == True, 0)
    mask_proper = mask_proper.where(mask_proper == False, 1)

    #Drop region dimension
    mask_proper = mask_proper.drop('region')

    #Remove standard_name = 'region'
    mask_proper.attrs.pop('standard_name', None)

    #Rename mask variable
    mask_proper.name = name_region

    #Rename latitude and longitude dimensions
    mask_proper = mask_proper.rename({'lat': 'latitude', 'lon': 'longitude'})

    #Set latitude and longitude from the reference file
    mask_proper = mask_proper.assign_coords(latitude=latitude, longitude=longitude)

    #Output
    #Export
    if export == True:
        if not os.path.exists(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"):
                os.makedirs(f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed")

        outdir = f"{rootdir}/{EXPVER_c}/{datatype.upper()}/Processed"

        #Save
        with ProgressBar():
            mask_proper.to_netcdf(f'{outdir}/{name_region}_{datatype}_{EXPVER_c}.nc',engine='h5netcdf')
         
        print(f"Created {name_region}_{datatype}_{EXPVER_c}.nc in {outdir}")

    else:
        return mask_proper
    
    
    
    
    
    
    
    
    

