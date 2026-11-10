'''
IFS processing script

Author: Vincent de Feiter (vincent.defeiter@wur.nl)

Version: 19 October, 2026
'''
#%%
#------------------------------------------------------------------------------------------------------------
#                                   I M P O R T    M O D U L E S 
#------------------------------------------------------------------------------------------------------------

#Processing
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import glob
import re
import dask
from dask.diagnostics import ProgressBar
from scipy.interpolate import interp1d

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

import sys
path_snellius = '/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/code-utils'
path_eddy = '/home/vincent/EU-CATRINE-WP5_6/Testbed - Amazon/code-utils'

sys.path.append(path_eddy)

import IFS_processing_functions 
import importlib
importlib.reload(IFS_processing_functions)

#%%
#------------------------------------------------------------------------------------------------------------
#                                   P A T H S  &  S E T T I N G S 
#------------------------------------------------------------------------------------------------------------

#Rootdir IFS
rootdir = "/gpfs/work4/0/meso/vincent/Amazon/large_scale"
model_def = "/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/Code - IFS/L137 def.csv"


#%%

#------------------------------------------------------------------------------------------------------------
#                                               R E F E R E N C E 
#------------------------------------------------------------------------------------------------------------

#Dictionaries
dict_model = IFS_processing_functions.dict_model
dict_surface = IFS_processing_functions.dict_surface

#Location
x_ATTO = -59.0055
y_ATTO = -2.1458

#Datetime
SCu_list=['2022-08-09','2022-08-10','2022-08-11','2022-08-15','2022-08-17','2022-08-18']
DeepCu_list=['2022-08-12','2022-08-13','2022-08-14','2022-08-16']
date_list_general = ['2022-08-09','2022-08-10','2022-08-11','2022-08-12','2022-08-13','2022-08-14','2022-08-15','2022-08-16','2022-08-17','2022-08-18']

delta_time_hr = -4

#z DALES
ds_DALES = xr.open_dataset("/home/vdefeiter/CloudRoots_DALES/ShCu/Output/R034/profiles.034.nc")
z_DALES = ds_DALES.zt.values + 40
ds_DALES.close()

#z RADIOSONDE (does also include tower data!)
df_radiosonde = pd.read_csv("/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/Reference files/20220809_1000UTC.csv")
z_ref = list(df_radiosonde.z.values)

z_radiosonde = z_ref[:17] + list(np.arange(320,8200,50))

#Aircraft 1
aircraft_1 = pd.read_csv("/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/Reference files/Aircraft_CloudRoots_flight1_profile.txt")
z_aircraft_1 = aircraft_1['alt'].values[1:].astype(float) - 120
z_aircraft_1 = np.sort(z_aircraft_1)

#Aicraft 2
aircraft_2 = pd.read_csv("/home/vdefeiter/EU-CATRINE-WP5_6/Testbed - Amazon/Reference files/Aircraft_CloudRoots_flight2_profile.txt")
z_aircraft_2 = aircraft_2['alt'].values[1:].astype(float) - 120
z_aircraft_2 = np.sort(z_aircraft_2)

#CO2 heights
heights_ghg = [24, 38, 53, 79, 321]

#Fixed range
z_range = np.arange(0,8050,50)

dates = ['20220808','20220809','20220810','20220811','20220812','20220813','20220814','20220815','20220816','20220817','20220818','20220819','20220820']
# dates = ['20220809','20220810','20220811','20220812','20220813']
# dates = ['20220814','20220815','20220816','20220817','20220818']
# dates = ['20220820']
#dates = ['20220819','20220820']

#Variables ml pre-processing
variables_ml = ['co2','t','q','u','v','w','dmfa','umfa']
variables_ml_itr2 = ['co2','t','q','u','v','w','dmfa','umfa','p212009','p212010','p212021','p212022','p212033','p212034'] 

#Variables interpolate to given heights
variables_ml_interpolate = ['co2','ws','theta','wdir','q','t','u','v','w','density','p']
variables_ml_interpolate_itr2 = ['co2','ws','theta','wdir','q','t','dco2dt_dyn','dco2dt_diff','dco2dt_conv','co2flx_dyn','co2flx_diff','co2flx_conv','u','v','w','density','p','dmfa','umfa']

#Variables sfc
variables_sfc = ['blh','gppbfas','cbh','cp','z','hcct','hcc','lsm','lsp','lai_lv','lai_hv','lcc','mcc','mucape','mucin','slhf','strd','str','ssr','sp','sr','sshf','ssrd','tcc','tp','aco2nee','aco2gpp','aco2rec']


#%%
#------------------------------------------------------------------------------------------------------------
#                                               L O A D   D A T A   
#------------------------------------------------------------------------------------------------------------


#Resolution
resolutions = ['itr2'] #, 'ia63','iwpf'] #['ia61','ia62','ia63','itr2','iwpf'] #['ia63','itr2'] #

for EXPVER in resolutions: 
    
    print(f'Processing {EXPVER}')
    
    if "_LS" in EXPVER:
        EXPVER_c = EXPVER
        EXPVER = EXPVER.replace("_LS","")
    else:
        EXPVER_c = EXPVER
    
    #-------------------------------
    #      S U R F A C E 
    #-------------------------------
    
    # print(f'Processing sfc')
    
    # #Process complete dataset
    # IFS_processing_functions.process_sfc(rootdir, EXPVER_c, variables_sfc,dict_surface,after_suffix='complete')

    # #Region
    # IFS_processing_functions.region_mask(rootdir,f"{rootdir}/{EXPVER_c}/SFC/{EXPVER}_sfc_ll_20220808.nc",'Amazon',EXPVER_c,'sfc',export=True)

    # #Subset
    # for date in dates:
    #     print(f'Processing {date}')
    #     IFS_processing_functions.subsetter_km(rootdir,EXPVER_c,f"{rootdir}/{EXPVER_c}/SFC/Processed/{EXPVER}_sfc_complete_{date}.nc",'sfc',x_ATTO,y_ATTO,100,f'subset_km_Amazon_{date}',export=True)
    
    # for date in dates:
    #     print(f'Processing {date}')
    #     IFS_processing_functions.subsetter_shapefile(rootdir,EXPVER_c,f"{rootdir}/{EXPVER_c}/SFC/Processed/{EXPVER}_sfc_complete_{date}.nc",'sfc',f'{rootdir}/{EXPVER_c}/SFC/Processed/Amazon_sfc_{EXPVER_c}.nc',export=True,after_suffix=f'subset_Amazon_{date}')
        
    # print(f'Processing sfc complete')
    
    
    #-------------------------------
    #        M O D E L 
    #-------------------------------
    
    print(f'Processing ml')
    
    # #Process complete dataset
    # if EXPVER == 'itr2' or EXPVER == 'iwpf' or EXPVER == 'j0xh':
    #     IFS_processing_functions.process_ml(rootdir, EXPVER, model_def, variables_ml_itr2, dict_model,tendency_in_data=True,after_suffix='complete')
    # else:
    #     IFS_processing_functions.process_ml(rootdir, EXPVER, model_def, variables_ml, dict_model,tendency_in_data=False,after_suffix='complete')

    #Region
    #IFS_processing_functions.region_mask(rootdir,f"{rootdir}/{EXPVER}/ML/{EXPVER}_ml_ll_20220808.nc",'Amazon',EXPVER,'ml',export=True)
    
    #Subset
    # for date in dates:
    #     print(f'Processing {date}')
    #     IFS_processing_functions.subsetter_km(rootdir,EXPVER,f"{rootdir}/{EXPVER}/ML/Processed/{EXPVER}_ml_complete_{date}.nc",'ml',x_ATTO,y_ATTO,100,f'subset_km_Amazon_{date}',export=True)

    # for date in dates:
    #     print(f'Processing {date}')
    #     IFS_processing_functions.subsetter_shapefile(rootdir,EXPVER,f"{rootdir}/{EXPVER}/ML/Processed/{EXPVER}_ml_complete_{date}.nc",'ml',f'{rootdir}/{EXPVER}/ML/Processed/Amazon_ml_{EXPVER}.nc',export=True,after_suffix=f'subset_Amazon_{date}')
    
   
    if EXPVER == 'itr2' or EXPVER == 'iwpf' or EXPVER == 'j0xh':
        
        for date in dates: 
            
            print(f'Processing {date}')
            
            #Radiosonde (including the tower data)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate_itr2,f'subset_km_Amazon_{date}',f'subset_km_Amazon_z_{date}',z_radiosonde,export=True)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate_itr2,f'subset_Amazon_{date}',f'subset_Amazon_z_{date}',z_radiosonde,export=True)

            #CO2 heights
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate_itr2,f'subset_km_Amazon_{date}',f'subset_km_Amazon_ghg_{date}',heights_ghg,export=True)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate_itr2,f'subset_Amazon_{date}',f'subset_Amazon_ghg_{date}',heights_ghg,export=True)

    else:
        
         for date in dates: 

            #Radiosonde (including the tower data)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate,f'subset_km_Amazon_{date}',f'subset_km_Amazon_z_{date}',z_radiosonde,export=True)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate,f'subset_Amazon_{date}',f'subset_Amazon_z_{date}',z_radiosonde,export=True)

            #CO2 heights
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate,f'subset_km_Amazon_{date}',f'subset_km_Amazon_ghg_{date}',heights_ghg,export=True)
            IFS_processing_functions.interpolate_to_heights(rootdir,EXPVER,variables_ml_interpolate,f'subset_Amazon_{date}',f'subset_Amazon_ghg_{date}',heights_ghg,export=True)
             
            print(f'Processing {date}')
           
    print(f'Processing ml complete')
