'''
EU-CATRINE Testbed | Amazon

This testbed includes the following for OBS, DALES, CLASS, IFS at three resolutions (28, 9 and 4.4 km):
> Timeseries of meteorological and surface variables at distinct heights. 
> Vertical profiles of meteorological variables in the lower troposphere. 
> Statistical quantities.
> Diagnostics (e.g., DXR, exchange coefficients etc).

The complete testbed is elaborated in the "Technical Report" document for further reference. 

Author: Vincent de Feiter (vincent.defeiter@wur.nl)

Version: 11 November, 2026

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

#Visualisation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms as transforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from matplotlib import patheffects
from matplotlib import rcParams
import matplotlib.patheffects as path_effects
import seaborn as sns
from itertools import combinations
import matplotlib.dates as mdates

import sys
sys.path.append('/home/vdefeiter/EUCATRINE_WP5_6/code-utils')

import testbed_functions
import CLASS
from CLASS import *


#%%
#------------------------------------------------------------------------------------------------------------
#                      S E T   V I S U A L I S A T I O N   P A R A M E T E R S
#------------------------------------------------------------------------------------------------------------

rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 15,
    "axes.labelsize": 15, "legend.fontsize": 15, "axes.titlepad": 20, "axes.labelpad": 20,
    "xtick.major.pad": 5, "ytick.major.pad": 5, "axes.edgecolor": 'k', "axes.linewidth": 1,
    "xtick.major.width": .8, "ytick.major.width": .8, "xtick.minor.width": .8, "ytick.minor.width": .8,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 4, "ytick.minor.size": 4
})
axiswidth = 0.8

#%%
#------------------------------------------------------------------------------------------------------------
#                                   P A T H S  &  S E T T I N G S 
#------------------------------------------------------------------------------------------------------------

#Directory where DALES data is stored
rootdir_DALES_034 = '/home/vdefeiter/CloudRoots_DALES/ShCu/Output/R034/'
rootdir_DALES_035 = '/home/vdefeiter/CloudRoots_DALES/ShCu/Output/R035/'

#Directory where OBS data is stored 
rootdir_OBS = "/home/vdefeiter/PhD/General/Reference files"

#Location where IFS is stored
rootdir_IFS = '/gpfs/work4/0/meso/vincent/Amazon'

#Location of code-utils
code_utils = '/home/vdefeiter/EUCATRINE_WP5_6/Testbed - Amazon/code-utils'

#Output directory
outdir = "/home/vdefeiter/EUCATRINE_WP5_6/Results"


#------------------------------------------------------------------------------------------

#Location ATTO
x_ATTO = -59.0055
y_ATTO = -2.1458

zmax = 3500

#Reference time arrays
SCu_list=['2022-08-09','2022-08-10','2022-08-11','2022-08-15','2022-08-17','2022-08-18']
DeepCu_list=['2022-08-12','2022-08-13','2022-08-14','2022-08-16']
date_list_general = ['2022-08-09','2022-08-10','2022-08-11','2022-08-12','2022-08-13','2022-08-14','2022-08-15','2022-08-16','2022-08-17','2022-08-18']
date_range = pd.date_range("2022-08-09","2022-08-18")

#Specify heights
ATTO_levels = [43, 100, 127, 151, 172, 196, 223, 247, 298] #dropped 316
INSTANT_levels = [5, 15, 25, 35, 50, 81]
heights_ghg = [24, 38, 53, 79, 321]

ATTO_levels = np.array(ATTO_levels)
INSTANT_levels = np.array(INSTANT_levels)

all_heights = np.sort(np.concatenate((ATTO_levels, INSTANT_levels), axis=0))


#------------------------------------------------------------------------------------------
#%%

#Plotting switches

#General
plot_timeseries = True
plot_profiles = True
plot_statistics = True
plot_diagnostics = True

plot_SCu = True
plot_DeepCu = True
plot_all_days = True

#What to include
plot_OBS = True
plot_DALES = True
plot_CLASS = True
plot_IFS = True

#View
OBS_std = True
IFS_std = False

#%%
#------------------------------------------------------------------------------------------------------------
#                                       L O A D   D A T A  - D A L E S  
#------------------------------------------------------------------------------------------------------------

#Clouds
profiles = xr.open_dataset(f'{rootdir_DALES_034}/profiles.034.nc',engine='h5netcdf')
tmser = xr.open_dataset(f'{rootdir_DALES_034}/tmser.034.nc',engine='h5netcdf')

#No Clouds
profiles_no_cloud = xr.open_dataset(f'{rootdir_DALES_035}profiles.035.nc',engine='h5netcdf')
tmser_no_cloud = xr.open_dataset(f'{rootdir_DALES_035}/tmser.035.nc',engine='h5netcdf')

#Profiles
DALES_profiles = testbed_functions.DALES_profiler(profiles)
DALES_profiles_no_clouds = testbed_functions.DALES_profiler(profiles_no_cloud)

#ABL 1D
DALES_abl_1D = testbed_functions.DALES_abl_1D(profiles)
DALES_abl_1D_no_clouds = testbed_functions.DALES_abl_1D(profiles_no_cloud)

#tmser
DALES_tmser = testbed_functions.DALES_tmsers(xr.open_dataset(f'{rootdir_DALES_034}tmser.034.nc',engine='h5netcdf'),open(f'{rootdir_DALES_034}tmlsm.034'),profiles,z_correction=40)
DALES_tmser_no_clouds = testbed_functions.DALES_tmsers(xr.open_dataset(f'{rootdir_DALES_035}tmser.035.nc',engine='h5netcdf'),open(f'{rootdir_DALES_035}tmlsm.035'),profiles_no_cloud,z_correction=40)

#Reference
z_DALES = DALES_profiles['zt']

#%%
#------------------------------------------------------------------------------------------------------------
#                                       L O A D   D A T A  -   C L A S S  (ShCu only)
#------------------------------------------------------------------------------------------------------------

#SCu
SCu_input = model_input()


#initial values for the 10th:

SCu_input.dt         = 10.       # time step [s]
SCu_input.runtime    = (12*3600)+10     # total run time [s]
SCu_input.hjump2     = 400.      # set if there is second theta & q jump; default = -1 !

# mixed-layer input
SCu_input.sw_ml      = True      # mixed-layer model switch
SCu_input.sw_shearwe = False     # shear growth mixed-layer switch !IvdLL: switched off, not sure why? 
SCu_input.sw_fixft   = False     # Fix the free-troposphere switch
SCu_input.h          = 385   # Initial ABL height [m] [4]
SCu_input.Ps         = 100000.0  # surface pressure [Pa] [3]
SCu_input.divU       = 0#1e-5           # horizontal large-scale divergence of wind [s-1]
SCu_input.fc         = -5.45*1e-6  # Coriolis parameter [m s-1]

SCu_input.theta      = 301     # initial mixed-layer potential temperature [K] [3]
SCu_input.dtheta     = 0.9       # initial temperature jump at h [K] [4]
SCu_input.gammatheta =  3.8e-3   # residual layer potential temperature lapse rate [K m-1] [4]
SCu_input.gammatheta2 = 3.8e-3    # free atmosphere potential temperature lapse rate [K m-1] 
SCu_input.advtheta   = 0.0      # advection of heat [K s-1], 
SCu_input.beta       = 0.2        # entrainment ratio for virtual heat 
SCu_input.wtheta     = 0.1      # surface kinematic heat flux [K m s-1] [3]

SCu_input.q          = 0.01629278    # initial mixed-layer specific humidity [kg kg-1] [3]
SCu_input.dq         = -0.0008     # initial specific humidity jump at h [kg kg-1] [*]
SCu_input.gammaq     = -3.3*1e-6   # residual layer specific humidity lapse rate [kg kg-1 m-1] [4]
SCu_input.gammaq2    = -3.3*1e-6   # free atmosphere specific humidity lapse rate [kg kg-1 m-1] [4]
SCu_input.advq       = 0.0       # advection of moisture [kg kg-1 s-1] 
SCu_input.wq         = 0   # surface kinematic moisture flux [kg kg-1 m s-1] [3]

SCu_input.CO2        = 445     # initial mixed-layer CO2 [ppm] [3]
SCu_input.dCO2       = -26 #-19.28 -5 # initial CO2 jump at h [ppm] [3]    # = B10 - (19,28+5)*1000
SCu_input.gammaCO2   = 0        # free atmosphere CO2 lapse rate [ppm m-1] 
# SCu_input.dCO22      = 0       # jump to free atmosphere CO2 lapse rate [ppm m-1]  (jump1-jump2)
SCu_input.advCO2     = 0         # advection of CO2 [ppm s-1] 
SCu_input.wCO2       = 0       # surface kinematic CO2 flux [ppm m s-1] [3]

SCu_input.O2         = -190       # initial mixed-layer O2 [ppm] [3]
SCu_input.dO2        = 18.54      # initial O2 jump at h [ppm] [3]
SCu_input.gammaO2    = 0          # free atmosphere O2 lapse rate [ppm m-1]
# SCu_input.dO22       = 0.         # jump to free atmosphere CO2 lapse rate [ppm m-1] (jump1-jump2)
SCu_input.advO2      = 0          # advection of O2 [ppm s-1] 
SCu_input.wO2        = 0.05       # surface kinematic O2 flux [ppm m s-1] [3]

SCu_input.ER_A      = -0.96       # Exchange ratio for assimilation [ppm ppm-1] 
SCu_input.ER_R      = -1.03       # Exchange ratio for respiration [ppm ppm-1] 

# Initialize isotopologues
SCu_input.sw_isot    = False      # isotopologues switch
## C13O2
# Version 1
#SCu_input.C13O2      = 4847.03818         # 13CO2 CBL value [ppb]
#SCu_input.dC13O2     = -405.70134  # FT jump of 13CO2  [ppb]
# Version 2
#SCu_input.C13O2      = 4848.9633         # 13CO2 CBL value [ppb]
#SCu_input.dC13O2     = -408.38794  # FT jump of 13CO2  [ppb]
# Version 3
SCu_input.C13O2      = 4872#4794.004         # 13CO2 CBL value [ppb]
SCu_input.dC13O2     = -260#-206.497#-200#-204.033         # FT jump of 13CO2  [ppb]
## H2O18
SCu_input.H2O18      = 17982.9       # H2O18 CBL value [ppb]
SCu_input.dH2O18     = -318.9        # FT jump of H2O18 [ppb]
## COO18
SCu_input.C18OO      = 1850.41994 #1619.8        # COO18 CBL value [ppb]
SCu_input.dC18OO     = -155.86488         # FT jump of COO18 [ppb]

SCu_input.sw_wind    = True      # prognostic wind switch, when False: 
SCu_input.u          = -1      # initial mixed-layer u-wind speed [m s-1] [CHECK THIS]
SCu_input.du         = -2.14        # initial u-wind jump at h [m s-1] 
SCu_input.gammau     = 0.0007788067   # free atmosphere u-wind speed lapse rate [s-1] 
SCu_input.advu       = 0.        # advection of u-wind [m s-2] [3]

SCu_input.v          = -2        # initial mixed-layer v-wind speed [m s-1] 
SCu_input.dv         = -0.85        # initial v-wind jump at h [m s-1] 
SCu_input.gammav     = -0.00030481        # free atmosphere v-wind speed lapse rate [s-1] 
SCu_input.advv       = 0.        # advection of v-wind [m s-2] [3]

SCu_input.sw_sl      = True      # surface layer switch
SCu_input.ustar      = 0.10297998      # surface friction velocity [m s-1] [3]
SCu_input.z0m        = 0.5        # roughness length for momentum [m] [2]
SCu_input.z0h        = 0.1        # roughness length for scalars [m] [2]

SCu_input.sw_rad     = True      # radiation switch
SCu_input.lat        = -2.14     # latitude [deg] 
SCu_input.lon        = -59.00     # longitude [deg] 
SCu_input.doy        = 226      # day of the year [-] : Adjusted for te 18th of may in Loobos
SCu_input.tstart     = 10        # time of the day [h UTC] # (7 LT)
SCu_input.cc         = 0.0       # cloud cover fraction [-]
SCu_input.Q          = 826.18      # net radiation [W m-2] 
SCu_input.dFz        = 0.        # cloud top radiative divergence [W m-2] 

SCu_input.sw_ls      = True      # land surface switch
SCu_input.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
SCu_input.wg         = 0.3     # volumetric water content top soil layer [m3 m-3] [3]
SCu_input.w2         = 0.3 #0.3    # volumetric water content deeper soil layer [m3 m-3] [3]
SCu_input.cveg       = 0.9      # vegetation fraction [-] 
SCu_input.Tsoil      = 298.59 #298   # temperature top soil layer [K] [3]
SCu_input.T2         = 298.7 #298.7 #300.30  # temperature deeper soil layer [K]  [3]
SCu_input.a          = 0.219    # Clapp and Hornberger retention curve parameter a From CLASS [Table 10.7, Sand] [2]
SCu_input.b          = 4.9     # Clapp and Hornberger retention curve parameter b From CLASS [Table 10.7, Sand] [2]
SCu_input.p          = 4.        # Clapp and Hornberger retention curve parameter c From CLASS [Table 10.7, Sand] [2]
SCu_input.CGsat      = 3.56e-6  # saturated soil conductivity for heat [2]

SCu_input.wsat       = 0.5     # saturated volumetric water content ECMWF config [-] 
SCu_input.wfc        = 0.4      # volumetric water content field capacity [-] 
SCu_input.wwilt      = 0.171     # volumetric water content wilting point [-] 

SCu_input.C1sat      = 0.132    # Coefficient force term moisture [-] [2]
SCu_input.C2ref      = 1.8      # Coefficient restore term moisture [-] [2]

SCu_input.LAI        = 5      # leaf area index [-] [3] 
SCu_input.gD         = 0.03      # correction factor transpiration for VPD [-] only for high vegetation, [2]
SCu_input.rsmin      = 175      # minimum resistance transpiration  [s m-1], [table 10.6] [2]
SCu_input.rssoilmin  = 50       # minimun resistance soil evaporation [s m-1], [2]
SCu_input.alpha      = 0.12     # surface albedo [-] [3]

SCu_input.Ts         = 298.96 #298 #296      # initial surface temperature [K] [3]

SCu_input.Wmax       = 0.0002   # thickness of water layer on wet vegetation [m], 
SCu_input.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m] 

SCu_input.Lambda     = 40         # thermal conductivity skin layer [-] [2]

SCu_input.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

SCu_input.sw_cu      = True     # Cumulus parameterization switch 
SCu_input.dz_h       = 193      # Transition layer thickness [m]

# A-gs (these are the standard values of CLASS):
# Plant type:       -C3-     -C4-
SCu_input.CO2comp298 =  [68.5,    4.3    ]   # CO2 compensation concentration [mg m-3] [/] wrong units
SCu_input.Q10CO2     =  [1.5,     1.5    ]   # function parameter to calculate CO2 compensation concentration [-]
SCu_input.gm298      =  [2,        17.5   ]   # C3: 7 mesophyill conductance at 298 K [mm s-1] #Kim
SCu_input.Ammax298   =  [1.83,     1.7    ]   # C3: 2.2, CO2 maximal primary productivity [mg m-2 s-1]
SCu_input.Q10gm      =  [2.0,     2.0    ]   # function parameter to calculate mesophyll conductance [-]
SCu_input.T1gm       =  [278.,    286.   ]   # reference temperature to calculate mesophyll conductance gm [K]
SCu_input.T2gm       =  [301.,    309.   ]   # reference temperature to calculate mesophyll conductance gm [K]
SCu_input.Q10Am      =  [2.0,     2.0    ]   # function parameter to calculate maximal primary profuctivity Ammax
SCu_input.T1Am       =  [281.,    286.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
SCu_input.T2Am       =  [311.,    311.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
SCu_input.f0         =  [0.89,    0.85   ]   # maximum value Cfrac [-] #Kim
SCu_input.ad         =  [0.07,    0.15   ]   # regression coefficient to calculate Cfrac [kPa-1]
SCu_input.alpha0     =  [0.017,   0.014  ]   # initial low light conditions [mg J-1]
SCu_input.Kx         =  [0.7,     0.7    ]   # extinction coefficient PAR [-]
SCu_input.gmin       =  [0.25e-3, 0.25e-3]   # cuticular (minimum) conductance [mm s-1] [m s-1] wrong units
SCu_input.R10        = 0.1

SCu_run = model(SCu_input)
SCu_run.run()

# Convert the dictionary to a JSON string
json_string = json.dumps(SCu_input.__dict__)

if not os.path.exists('/home/vdefeiter/EUCATRINE_WP5_6/code-utils/CLASS_SCu_input.txt'):
    with open('/home/vdefeiter/EUCATRINE_WP5_6/code-utils/CLASS_SCu_input.txt', 'w') as f:
        f.write(json_string)

SCu_input.sw_cu      = False     # Cumulus parameterization switch 
SCu_run_no_clouds = model(SCu_input)
SCu_run_no_clouds.run()

#%%
#------------------------------------------------------------------------------------------------------------
#                                  L O A D  D A T A  -  O B S 
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                       T O W E R   D A T A 
#------------------------------------------------------------------------------------------------------------------------------------------------------

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
  
#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                       R A D I O S O N D E S 
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Process radiosondes
SCu_list_s = [20220809, 20220810, 20220811, 20220815, 20220817, 20220818]
DeepCu_list_s = [20220812, 20220813, 20220814, 20220816]
SCu_list_vertical = ['20220809','20220810','20220811','20220815','20220817','20220818']
DeepCu_list_vertical = ['20220812','20220813','20220814','20220816']

time_list = np.array([1000, 1300, 1600, 1900, 2200])
time_list_str = ['6', '9', '12', '15', '18']

SCu_values_LCL, SCu_values_LFC, SCu_values_EL, SCu_values_cape, SCu_values_cin, SCu_values_h = testbed_functions.process_values_sonde(SCu_list_s,time_list,f'{rootdir_OBS}/Vertical Profiles/' + 'Timeseries Reference.csv')
DeepCu_values_LCL, DeepCu_values_LFC, DeepCu_values_EL, DeepCu_values_cape, DeepCu_values_cin, DeepCu_values_h = testbed_functions.process_values_sonde(DeepCu_list_s,time_list,f'{rootdir_OBS}/Vertical Profiles/' + 'Timeseries Reference.csv')

SCu_values_z_ref, SCu_values_p_ref, SCu_values_theta_ref, SCu_values_q_ref, SCu_values_RH_ref, SCu_values_T_ref, SCu_values_Td_ref, SCu_values_wind_u_ref, SCu_values_wind_v_ref, SCu_values_wind_speed_ref, SCu_values_wind_dir_ref = testbed_functions.append_values_radiosondes(SCu_list_vertical, time_list, 'SCu', f'{rootdir_OBS}/Vertical Profiles/', ref=True)
DeepCu_values_z_ref, DeepCu_values_p_ref, DeepCu_values_theta_ref, DeepCu_values_q_ref, DeepCu_values_RH_ref, DeepCu_values_T_ref, DeepCu_values_Td_ref, DeepCu_values_wind_u_ref, DeepCu_values_wind_v_ref, DeepCu_values_wind_speed_ref, DeepCu_values_wind_dir_ref = testbed_functions.append_values_radiosondes(DeepCu_list_vertical, time_list, 'DeepCu', f'{rootdir_OBS}/Vertical Profiles/', ref=True)
SCu_values_z, SCu_values_p, SCu_values_theta, SCu_values_q, SCu_values_RH, SCu_values_T, SCu_values_Td, SCu_values_wind_u, SCu_values_wind_v, SCu_values_wind_speed, SCu_values_wind_dir = testbed_functions.append_values_radiosondes(SCu_list_vertical, time_list, 'SCu', f'{rootdir_OBS}/Vertical Profiles/', ref=False)
DeepCu_values_z, DeepCu_values_p, DeepCu_values_theta, DeepCu_values_q, DeepCu_values_RH, DeepCu_values_T, DeepCu_values_Td, DeepCu_values_wind_u, DeepCu_values_wind_v, DeepCu_values_wind_speed, DeepCu_values_wind_dir = testbed_functions.append_values_radiosondes(DeepCu_list_vertical, time_list, 'DeepCu', f'{rootdir_OBS}/Vertical Profiles/', ref=False)

z_radiosonde = SCu_values_z_ref[0][0]


#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                            B O U N D A R Y   L A Y E R   &  C L O U D S  
#------------------------------------------------------------------------------------------------------------------------------------------------------

# Load ABL Information
df_ABL = pd.read_csv(f'{rootdir_OBS}/ABL_SCu_DeepCu_aggregates.csv')
abl_SCu_mean = df_ABL['SCu_mean'].values
abl_SCu_std = df_ABL['SCu_std'].values
abl_DeepCu_mean = df_ABL['DeepCu_mean'].values
abl_DeepCu_std = df_ABL['DeepCu_std'].values

df_ABL_total = pd.read_csv(f'{rootdir_OBS}/ABL_filtered.csv')
abl_daily = df_ABL_total['blh'].values

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                               R E S P I R A T I O N  &  G P P 
#------------------------------------------------------------------------------------------------------------------------------------------------------

# Process soil respiration data (single function)
Resp_dataset = testbed_functions.process_soil_respiration(
    filepath=f'{rootdir_OBS}/Soil respiration ATTO 5-31 august 2022.xlsx',
    date_list_general=date_list_general,
    date_range=date_range
)

# Aggregate Respiration profiles (separate function)
SCu_Resp_mean, SCu_Resp_std = testbed_functions.aggregate_case_mean_std(Resp_dataset, 'Resp', SCu_list)
DeepCu_Resp_mean, DeepCu_Resp_std = testbed_functions.aggregate_case_mean_std(Resp_dataset, 'Resp', DeepCu_list)

# Calculate GPP (separate function)
daily_arrays = testbed_functions.daily_range(date_range, '30min', end="18:00")
GPP_dataset = testbed_functions.calculate_GPP_dataset(Resp_dataset, daily_tower['NEE'][5], daily_arrays)

# Aggregate GPP profiles (separate function)
SCu_GPP_mean, SCu_GPP_std = testbed_functions.aggregate_case_mean_std(GPP_dataset, 'GPP', SCu_list)
DeepCu_GPP_mean, DeepCu_GPP_std = testbed_functions.aggregate_case_mean_std(GPP_dataset, 'GPP', DeepCu_list)

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                   R A D I A T I O N   P R O C E S S I N G
#------------------------------------------------------------------------------------------------------------------------------------------------------

SCu_rad_mean, SCu_rad_std, DeepCu_rad_mean, DeepCu_rad_std, daily_rad = testbed_functions.process_radiation_data(
    f'{rootdir_OBS}/RadiationTemp_CloudRoots.csv',
    date_list_general,
    SCu_list,
    DeepCu_list
)

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                  G A S E S   P R O C E S S I N G
#------------------------------------------------------------------------------------------------------------------------------------------------------

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

# Aircraft data
flight1_data = testbed_functions.load_aircraft_profile(f"{rootdir_OBS}/Aircraft_CloudRoots_flight1_profile.txt")
flight2_data = testbed_functions.load_aircraft_profile(f"{rootdir_OBS}/Aircraft_CloudRoots_flight2_profile.txt")
#%%
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#                                V I S U A L I S E - E C M W F  I F S  -  R E G I M E  C L A S S I F I C A T I O N  
#-------------------------------------------------------------------------------------------------------------------------------------------------------

x_IFS = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="60min").time)

resolutions = ['28 km', '9 km', '4.4 km']
experiments = ['ia61', 'ia62', 'ia63']

a = 0
for res in experiments: 

    ds_sfc = xr.open_mfdataset(f"{rootdir_IFS}/{res}/SFC/Processed/{res}_sfc_subset_km_Amazon_2022*.nc").mean(dim=['latitude', 'longitude'])
    ds_sfc['time'] = pd.to_datetime(ds_sfc['time'].values) + timedelta(hours=-4)
    
    fig, ax = plt.subplots(5,2,figsize=(11.69*1.2,8.27*2),dpi=300) 
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = ax.flatten()
    
    count = 1
    for date in date_list_general: 
        
        # date = date_list_general[3]
        
        ds_sfc_sub = ds_sfc.sel(time=date)
        
        #Sel data
        cp_sub = ds_sfc_sub['cp'].values*1000
        blh_sub = ds_sfc_sub['blh'].values
        cbh_sub = ds_sfc_sub['cbh'].values
        hcct_sub = ds_sfc_sub['hcct'].values
        
        twin_axes = locals()[f'ax{count}'].twinx()
        
        #Precipitation
        twin_axes.bar(ds_sfc_sub.time, cp_sub, color='tab:blue',width=0.03, alpha=0.3)
        
        #BLH
        locals()[f'ax{count}'].plot(ds_sfc_sub.time, blh_sub, color='k',label='h')
        
        #CLOUD LAYER
        locals()[f'ax{count}'].plot(ds_sfc_sub.time, cbh_sub, color='tab:gray',label='cloud base')
        locals()[f'ax{count}'].plot(ds_sfc_sub.time, hcct_sub, color='tab:gray', linestyle='--',label='convective cloud top')
        
        locals()[f'ax{count}'].fill_between(ds_sfc_sub.time, cbh_sub, hcct_sub, color='tab:gray', alpha=0.3,label='cloud layer')
        
        locals()[f'ax{count}'].bar([],[], color='tab:blue',width=0.03, alpha=0.3,label='convective precipitation')
          
        #Detailing
        locals()[f'ax{count}'].grid(linestyle=':',color='tab:gray',alpha=0.3,linewidth=axiswidth*2)
        
        locals()[f'ax{count}'].set_xlim(ds_sfc_sub.time[0], ds_sfc_sub.time[-1])
        twin_axes.set_xlim(ds_sfc_sub.time[0], ds_sfc_sub.time[-1])
        
        locals()[f'ax{count}'].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        twin_axes.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        
        locals()[f'ax{count}'].set_ylim(0,6000)
        twin_axes.set_ylim(0,12)
        twin_axes.set_yticks(np.arange(0,14,2),np.arange(0,14,2))
        
        locals()[f'ax{count}'].tick_params(axis='x')
        locals()[f'ax{count}'].tick_params(axis='y')
        twin_axes.tick_params(axis='y')
        twin_axes.tick_params(axis='x')
        
        locals()[f'ax{count}'].xaxis.set_minor_locator(AutoMinorLocator())
        twin_axes.xaxis.set_minor_locator(AutoMinorLocator())
        locals()[f'ax{count}'].yaxis.set_minor_locator(AutoMinorLocator())
        twin_axes.yaxis.set_minor_locator(AutoMinorLocator())
        
        locals()[f'ax{count}'].tick_params(which='major')
        locals()[f'ax{count}'].tick_params(which='minor')
        twin_axes.tick_params(which='major')
        twin_axes.tick_params(which='minor')

        locals()[f'ax{count}'].set_xlabel('LT (hour)')
        locals()[f'ax{count}'].set_ylabel('z (m)')
        twin_axes.set_ylabel('(mm)')   
        
        #Add date
        if date in SCu_list:
            label = rf'{date} ($\mathbf{{ShCu}}$)'
        else:
            label = rf'{date} ($\mathbf{{ShDeep}}$)'

        locals()[f'ax{count}'].text(
            0.02, 1.15,
            label,
            transform=locals()[f'ax{count}'].transAxes,
            verticalalignment='top'
        )

        #Add label (a, b, c etc.)
        locals()[f'ax{count}'].text(-0.25, 1.1, f'$\mathbf{{({chr(96+count)})}}$', transform=locals()[f'ax{count}'].transAxes, fontweight='bold', verticalalignment='top')
        
        count = count + 1 
    
    plt.subplots_adjust(hspace=1,wspace=0.6)
    
    legend = ax1.legend(loc='lower left', bbox_to_anchor= (0.5, 1.4), ncol=3,
                                borderaxespad=0, frameon=False,markerscale=1,borderpad=0.7)
    
    #Figure title
    plt.suptitle(rf'$\mathbf{{ECMWF-IFS \:\: | \:\: {resolutions[a]}}}$', y=1.00, fontsize = plt.rcParams['font.size']*1.5)
    
    #Export
    if not os.path.exists(f"{outdir}/classifcation/"):
        os.makedirs(f"{outdir}/classifcation/")

    plt.savefig(f"{outdir}/classifcation/classification_{resolutions[a]}.pdf",bbox_inches='tight')   
    plt.savefig(f"{outdir}/classifcation/classification_{resolutions[a]}.jpg",bbox_inches='tight',dpi=300)
    
    plt.show()
    plt.close()
    
    a = a + 1 

#%%
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                  V I S U A L I S E  -  T I M E S E R I E S 
#-------------------------------------------------------------------------------------------------------------------------------------------------------

resolutions = {'28 km' : ['ia61','tab:purple'],
    '9 km' : ['ia62','tab:orange'],
    '4.4 km' : ['ia63','k'],
}

dict_var = {
    
    'Qnet' : {'label' : 'Q$_{\mathrm{net}}$ (W m$^{-2}$)',
            'step': 100, 
            'varname': 'qnet', 
            'datatype' : 'sfc',
            'multiplier': 1},

    'SWnet' : {'label' : 'SW$_{\mathrm{net}}$ (W m$^{-2}$)',
            'step': 200,
            'varname': 'ssr',
            'datatype' : 'sfc',
            'multiplier': 1},

    'LWnet' : {'label' : 'LW$_{\mathrm{net}}$ (W m$^{-2}$)',
            'step': 20,
            'varname': 'str',
            'datatype' : 'sfc',
            'multiplier': 1},

    'SWin' : {'label' : 'SW$_{\mathrm{in}}$ (W m$^{-2}$)',
            'step': 200,
            'varname': 'ssrd',
            'datatype' : 'sfc',
            'multiplier': 1},

    'SWout' : {'label' : 'SW$_{\mathrm{out}}$ (W m$^{-2}$)',
            'step': 25,
            'varname': 'ssru',
            'datatype' : 'sfc',
            'multiplier': -1},

    'LWin' : {'label' : 'LW$_{\mathrm{in}}$ (W m$^{-2}$)',
            'step': 20,
            'varname': 'strd',
            'datatype' : 'sfc',
            'multiplier': 1},

    'LWout' : {'label' : 'LW$_{\mathrm{out}}$ (W m$^{-2}$)',
            'step': 20,
            'varname': 'stru',
            'datatype' : 'sfc',
            'multiplier': -1},
    
    'H' : {'label' : 'H (W m$^{-2}$)',
            'step': 50, 
            'varname': 'sshf', 
            'datatype' : 'sfc',
            'multiplier': -1},
    
    'LE' : {'label' : 'LE (W m$^{-2}$)',
            'step': 100, 
            'varname': 'slhf', 
            'datatype' : 'sfc',
            'multiplier': -1},
    
    'EF' : {'label' : 'EF (-)',
            'step': 0.2,
            'varname': 'EF',
            'datatype' : 'sfc',
            'multiplier': 1},
    
    'NEE' : {'label' : r'NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',
            'step': 5, 
            'varname': 'aco2nee', 
            'datatype' : 'sfc',
            'multiplier': 1},

    'GPP' : {'label' : r'GPP ($\mu$mol m$^{-2}$ s$^{-1}$)',
            'step': 5, 
            'varname': 'aco2gpp', 
            'datatype' : 'sfc',
            'multiplier': -1},
    
    'Resp' : {'label' : r'RESP ($\mu$mol m$^{-2}$ s$^{-1}$)',
            'step': 2, 
            'varname': 'aco2rec', 
            'datatype' : 'sfc',
            'multiplier': 1},
    
    'theta' : {'label' : r'$\theta$ (K)',
            'step': 2,
            'varname': 'theta',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'q' : {'label' : 'q (kg kg$^{-1}$)',
            'step': 2,
            'varname': 'q',
            'datatype' : 'ml',
            'multiplier': 1},

    'CO2': {'label' : 'CO$_2$ (ppm)',
            'step': 50,
            'varname': 'co2',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'U': {'label' : 'U (m s$^{-1}$)',
            'step': 1,
            'varname': 'ws',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'WD': {'label' : 'WD (°)',
            'step': 90,
            'varname': 'wdir',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'hcc': {'label' : 'High Cloud Cover (-)',
                    'step': 0.2,
                    'varname': 'hcc',
                    'datatype' : 'sfc',
                    'multiplier': 1},
    
    'mcc': {'label' : 'Medium Cloud Cover (-)',
                    'step': 0.2,
                    'varname': 'mcc',
                    'datatype' : 'sfc',
                    'multiplier': 1},

    'lcc': {'label' : 'Low Cloud Cover (-)',
                    'step': 0.2,
                    'varname': 'lcc',
                    'datatype' : 'sfc',
                    'multiplier': 1},
    
    'mass flux' : {'label' : 'M (m s$^{-1}$)',
                'step': 0.1,
                'varname': 'mass_flux',
                'datatype' : 'sfc',   
                    'multiplier': 1},
    
    'h' : {'label' : 'z$_i$ (m)',
            'step': 500, 
            'varname': 'blh', 
            'datatype' : 'sfc',
            'multiplier': 1},
    
} 

vars_rad = ['Qnet','SWnet','SWin','SWout','LWin','LWout','LWnet']
vars_tower = ['H','LE','EF','NEE','GPP', 'Resp']
vars_tower_state = ['theta','q','CO2','U','WD']
vars_cloud_cover = ['hcc','mcc','lcc']

modes = ['SCu', 'DeepCu', 'daily']

if plot_timeseries == True:
    
    for mode in modes: 
            
        print(mode)
            
        for var in dict_var.keys():  
          
            
            print(var)
        
            #Define reference height based on variable type
            z_refs = (
                [75] if var in vars_rad else
                [50, 298] if var in vars_tower_state else
                [50] if var in vars_tower else
                [None]
            )
            
            #Go over each reference height
            for z_ref in z_refs: 
                print(f"z_ref: {z_ref} m")
                
                z_str_OBS = z_ref

                #---------------------------------------------- X - O B S E R V A T I O N S ---------------------------------------------------
                
                #SCu and DeepCu
                if mode != 'daily':        
                    time_freqs = {
                        **{v: '10min' for v in vars_rad},
                        **{v: '30min' for v in vars_tower},
                        **{v: '30min' for v in vars_tower_state if v != 'CO2'},
                        'CO2': '1min',
                        'h': '5min'
                    }

                    freq = time_freqs.get(var)
                    if freq:
                        x_OBS = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq=freq).time)
                        if var == 'h':
                            x_OBS = x_OBS[1:]
                            x_OBS_radiosonde = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="3h").time)

                    
                elif mode == 'daily':
                    daily_freqs = {
                        **{v: '10min' for v in vars_rad},
                        **{v: '30min' for v in vars_tower},
                        **{v: '30min' for v in vars_tower_state if v != 'CO2'},
                        'CO2': '1min',
                        'h': '5min',
                        **{v: '1h' for v in ['hcc','mcc','lcc']}
                    }

                    freq = daily_freqs.get(var)
                    if freq:
                        x_OBS = testbed_functions.daily_range(date_range, freq, end="18:00")
                        if var == 'h':
                            x_OBS = x_OBS[~x_OBS.strftime('%H.%M').isin(['06.00'])]

                    x_OBS = np.asarray(x_OBS, dtype="datetime64[ns]")

                #---------------------------------------------- Y - O B S E R V A T I O N S ---------------------------------------------------
                
                #Load SCu and DeepCu observations
                if mode != 'daily':
                    if var in vars_rad:
                        y_OBS, y_OBS_std = locals()[f'{mode}_rad_mean'][var], locals()[f'{mode}_rad_std'][var]

                    elif var in vars_tower and var not in ['GPP', 'Resp']:
                        idx = testbed_functions.find_closest(all_heights, z_ref)
                        y_OBS, y_OBS_std = locals()[f'{mode}_tower_mean'][var][idx], locals()[f'{mode}_tower_std'][var][idx]

                    elif var in vars_tower_state and var != 'CO2':
                        var_s = {'U': 'wind_speed', 'WD': 'wind_dir'}.get(var, var)
                        idx = testbed_functions.find_closest(all_heights, z_ref)
                        y_OBS, y_OBS_std = locals()[f'{mode}_tower_mean'][var_s][idx], locals()[f'{mode}_tower_std'][var_s][idx]

                    elif var in ['GPP', 'Resp']:
                        y_OBS, y_OBS_std = locals()[f'{mode}_{var}_mean'], locals()[f'{mode}_{var}_std']

                    elif var == 'CO2':
                        z_str = f"{heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]}m"
                        y_OBS, y_OBS_std = locals()[f'{mode}_ghg_mean']['CO2'][z_str], locals()[f'{mode}_ghg_std']['CO2'][z_str]
                        z_str_OBS = z_str

                    elif var == 'h':
                        y_OBS, y_OBS_std = locals()[f'abl_{mode}_mean'], locals()[f'abl_{mode}_std']
                        if mode == 'SCu':
                            y_OBS_radiosonde = np.nanmean(SCu_values_h, axis=0)
                        elif mode == 'DeepCu':
                            y_OBS_radiosonde = np.nanmean(DeepCu_values_h, axis=0)

                #Load daily observations
                elif mode == 'daily':
                    if var in vars_rad:
                        y_OBS = daily_rad[var]

                    elif var in vars_tower and var not in ['GPP', 'Resp']:
                        z_str_OBS = all_heights[testbed_functions.find_closest(all_heights, z_ref)]
                        y_OBS = daily_tower[var][testbed_functions.find_closest(all_heights, z_ref)]
                        z_str_OBS = z_str

                    elif var in vars_tower_state and var != 'CO2':
                        var_s = {'U': 'wind_speed', 'WD': 'wind_dir'}.get(var, var)
                        y_OBS = daily_tower[var_s][testbed_functions.find_closest(all_heights, z_ref)]
                        z_str_OBS = all_heights[testbed_functions.find_closest(all_heights, z_ref)]

                    elif var == 'GPP':
                        y_OBS = GPP_dataset.GPP.values

                    elif var == 'Resp':
                        y_OBS = Resp_dataset.Resp.values

                    elif var == 'CO2':
                        z_str = f"{heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]}m"
                        y_OBS = daily_ghg['CO2'][z_str]
                        z_str_OBS = z_str

                    elif var == 'h':
                        y_OBS = abl_daily

                #---------------------------------------------- M O D E L S ---------------------------------------------------
                
                #------------------------------------------------ DALES -------------------------------------------------------
                
                #x
                x_DALES = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="5min").time)[1:]
                
                #y
                if mode == 'SCu': 
                    if var in vars_rad or var in vars_tower_state:
                        idx = testbed_functions.find_closest(z_DALES, z_ref)
                        z_str_DALES = int(z_DALES[idx])
                        y_DALES = DALES_profiles[var][:, idx]
                        y_DALES_no_clouds = DALES_profiles_no_clouds[var][:, idx]

                    elif var in vars_tower:
                        y_DALES = DALES_tmser[var]
                        y_DALES_no_clouds = DALES_tmser_no_clouds[var]

                    elif var in ['hcc','mcc','lcc']:
                        y_DALES = DALES_tmser['cfrac']
                        y_DALES_no_clouds = [0] * len(DALES_tmser_no_clouds['cfrac'])

                    elif var == 'h':
                        y_DALES = DALES_tmser['abl_1D']
                        y_DALES_no_clouds = DALES_tmser_no_clouds['abl_1D']

                    elif var == 'mass flux':
                        idx = DALES_tmser['idx_1D']
                        idx_no_clouds = DALES_tmser_no_clouds['idx_1D']

                        ref = [np.nanmean(DALES_profiles['thetav'][:, :int(i)]) if not np.isnan(i) else np.nan for i in idx]
                        ref_no_clouds = [np.nanmean(DALES_profiles_no_clouds['thetav'][:, :int(i)]) if not np.isnan(i) else np.nan for i in idx_no_clouds]

                        DALES_wstar = (9.81 * np.array(DALES_tmser['abl_1D']) * np.array(DALES_profiles['wthetav']) / ref) ** (1. / 3.)
                        DALES_wstar_no_clouds = (9.81 * np.array(DALES_tmser_no_clouds['abl_1D']) * np.array(DALES_profiles_no_clouds['wthetav']) / ref_no_clouds) ** (1. / 3.)

                        y_DALES = DALES_wstar * np.array(DALES_tmser['cfrac']) / 2
                        y_DALES_no_clouds = [0] * len(DALES_wstar_no_clouds)

                #------------------------------------------------ CLASS -------------------------------------------------------
                
                #x
                x_CLASS = np.linspace(6,18,len(SCu_run.out.t))
                
                #y
                if mode == 'SCu':
                    if var == 'Qnet':
                        y_CLASS = SCu_run.out.Q
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Q
                    elif var == 'SWnet':
                        y_CLASS = SCu_run.out.Swin - SCu_run.out.Swout
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Swin - SCu_run_no_clouds.out.Swout
                    elif var == 'LWnet':
                        y_CLASS = SCu_run.out.Lwin - SCu_run.out.Lwout
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Lwin - SCu_run_no_clouds.out.Lwout
                    elif var == 'SWin':
                        y_CLASS = SCu_run.out.Swin
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Swin
                    elif var == 'SWout':
                        y_CLASS = SCu_run.out.Swout
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Swout
                    elif var == 'LWin':
                        y_CLASS = SCu_run.out.Lwin
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Lwin
                    elif var == 'LWout':
                        y_CLASS = SCu_run.out.Lwout
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.Lwout
                    elif var == 'H':
                        y_CLASS = SCu_run.out.H
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.H
                    elif var == 'LE':
                        y_CLASS = SCu_run.out.LE
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.LE
                    elif var == 'EF':
                        y_CLASS = SCu_run.out.LE / (SCu_run.out.H + SCu_run.out.LE)
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.LE / (SCu_run_no_clouds.out.H + SCu_run_no_clouds.out.LE)
                    elif var == 'NEE':
                        y_CLASS = SCu_run.out.wCO2 * 40.876
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.wCO2 * 40.876
                    elif var == 'GPP':
                        y_CLASS =  (SCu_run.out.wCO2R - SCu_run.out.wCO2) * 40.876
                        y_CLASS_no_clouds = (SCu_run_no_clouds.out.wCO2R - SCu_run_no_clouds.out.wCO2) * 40.876
                    elif var == 'Resp':
                        y_CLASS = SCu_run.out.wCO2R * 40.876
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.wCO2R * 40.876
                    elif var == 'theta':
                        y_CLASS = SCu_run.out.theta
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.theta
                    elif var == 'q':
                        y_CLASS = SCu_run.out.q*1000
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.q*1000
                    elif var == 'CO2':
                        y_CLASS = SCu_run.out.CO2
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.CO2
                    elif var == 'U':
                        y_CLASS = np.sqrt(SCu_run.out.u**2 + SCu_run.out.v**2)
                        y_CLASS_no_clouds = np.sqrt(SCu_run_no_clouds.out.u**2 + SCu_run_no_clouds.out.v**2)
                    elif var == 'WD':
                        y_CLASS = np.arctan2(SCu_run.out.v, SCu_run.out.u) * (180/np.pi) % 360
                        y_CLASS_no_clouds = np.arctan2(SCu_run_no_clouds.out.v, SCu_run_no_clouds.out.u) * (180/np.pi) % 360
                    elif var in ['hcc','mcc','lcc']:
                        y_CLASS = SCu_run.out.ac*2
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.ac*2
                    elif var == 'mass flux':
                        y_CLASS = SCu_run.out.M
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.M
                    elif var == 'h':
                        y_CLASS = SCu_run.out.h
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.h

                
                #----------------------------------------- V I S U A L I S E ------------------------------------------------    

                if not (var == 'mass flux' and mode in ['DeepCu', 'daily']):
                    
                    #Details
                    label = dict_var[var]['label']
                    step = dict_var[var]['step']
                    varname = dict_var[var]['varname']
                    datatype = dict_var[var]['datatype']
                    multiplier = dict_var[var]['multiplier']
                    
                    #Define a colour for the plot
                    colour = 'tab:red' if mode == 'SCu' else 'tab:blue'
                    
                    #Load figure
                    if mode == 'SCu' or mode == 'DeepCu':
                        fig, ax = plt.subplots(figsize=(8.27, 11.69/2),dpi=300)
                    elif mode == 'daily': 
                        fig, ax = plt.subplots(figsize=(8.27*1.6, 11.69/2),dpi=300)

                    #OBS
                    if mode != 'daily':
                        if var not in ['hcc','mcc','lcc','mass flux']:
                            
                            if var == 'NEE':
                                plot_label = f"OBS ShCu {str(z_str_OBS).split('m')[0]} m" if mode == 'SCu' else f"OBS ShDeep {str(z_str_OBS).split('m')[0]} m"
                            elif var in vars_tower or var=='h':
                                plot_label = f"OBS ShCu" if mode == 'SCu' else f"OBS ShDeep"
                            else:
                                plot_label = f"OBS ShCu {str(z_str_OBS).split('m')[0]} m" if mode == 'SCu' else f"OBS ShDeep {str(z_str_OBS).split('m')[0]} m"

                            ax.plot(x_OBS, y_OBS, linewidth=6.5, color=colour, ls="", marker='o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, alpha=0.2, label=plot_label)

                            if OBS_std == True:
                                ax.fill_between(x_OBS, y_OBS-y_OBS_std, y_OBS+y_OBS_std, color=colour, alpha=0.15)

                            if var == 'h': 
                                ax.plot(x_OBS_radiosonde,y_OBS_radiosonde,marker='^',markersize=6,markeredgecolor='k',markerfacecolor=colour,ls='',label='OBS ShCu (radiosonde)' if mode == 'SCu' else 'OBS ShDeep (radiosonde)')
                    
                    elif mode == 'daily':
                        if var not in ['hcc','mcc','lcc','mass flux']:
                            
                            if var == 'NEE':
                                plot_label = f"OBS {str(z_str_OBS).split('m')[0]} m"
                            elif var in vars_tower or var=='h':
                                plot_label = f"OBS"
                            else:
                                plot_label = f"OBS {str(z_str_OBS).split('m')[0]} m" 

                            ax.plot(x_OBS, y_OBS, linewidth=6.5, color='tab:gray', ls="", marker='o', markersize=5, markeredgecolor='k', markeredgewidth=0.5, alpha=0.3, label=plot_label)

                    
                    #CLASS
                    if plot_CLASS == True:
                        if mode == 'SCu': 
                            ax.plot(x_CLASS,y_CLASS,color='tab:blue',label='CLASS CLOUDS')
                            ax.plot(x_CLASS,y_CLASS_no_clouds,color='tab:blue',ls='--',label='CLASS NO CLOUDS')
                            
                    #DALES
                    if plot_DALES == True:
                        if mode == 'SCu': 
                            if var not in ['hcc','mcc','lcc','mass flux']:
                                
                                if var in vars_tower or var=='h' or var in vars_rad:
                                    plot_label = f"DALES"
                                    plot_label_no_clouds = f"DALES NO CLOUDS"
                                
                                else:
                                    plot_label = f"DALES {str(z_str_DALES).split('m')[0]} m"
                                    plot_label_no_clouds = f"DALES NO CLOUDS {str(z_str_DALES).split('m')[0]} m"

                                ax.plot(x_DALES,y_DALES,color='tab:green',label=plot_label)
                                ax.plot(x_DALES,y_DALES_no_clouds,color='tab:green',ls='--',label=plot_label_no_clouds)
                                
                            else:
                                ax.plot(x_DALES,y_DALES,color='tab:green',label='DALES CLOUDS')
                                ax.plot(x_DALES,y_DALES_no_clouds,color='tab:green',ls='--',label='DALES NO CLOUDS')

                    #IFS
                    IFS_data_y = []
                    x_IFS = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="60min").time) #if not specified differently below
                    if var != 'mass flux':
                        for key in resolutions.keys():
                            
                            res = resolutions[key][0]
                            IFS_colour = resolutions[key][1]
                            
                            if mode !='daily':
                                
                                if datatype == 'sfc':
                                    ds_base = xr.open_mfdataset(f"{rootdir_IFS}/{res}/SFC/Processed/{res}_sfc_subset_km_Amazon_2022*.nc",engine='h5netcdf').mean(dim=['latitude','longitude'])
                                    ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                    
                                    if mode == 'SCu':
                                        sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                                    else:
                                        sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                                    
                                    sub_mean = sub.groupby('time.hour').mean(dim='time')
                                    sub_std = sub.groupby('time.hour').std(dim='time')
                                    
                                    #Rename
                                    sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                                    sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                                    
                                    #ds
                                    ds = xr.merge([sub_mean, sub_std])
                                    ds = ds[sorted(ds.data_vars)]
                                    
                                    #Selection of hours
                                    ds =  ds.sel(hour=slice(6, 18))

                                else:
                                    if var == 'CO2':
                                        z_sel_CO2 = heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]
                                        
                                        ds_base = xr.open_mfdataset(f'{rootdir_IFS}/{res}/ML/Processed/{res}_ml_subset_z_ghg_2022*.nc',engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_sel_CO2)
                                        ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                        
                                        if mode == 'SCu':
                                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                                        else:
                                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                                        
                                        sub_mean = sub.groupby('time.hour').mean(dim='time')
                                        sub_std = sub.groupby('time.hour').std(dim='time')
                                        
                                        #Rename
                                        sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                                        sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                                        
                                        #ds
                                        ds = xr.merge([sub_mean, sub_std])
                                        ds = ds[sorted(ds.data_vars)]
                                        
                                        #Selection of hours
                                        ds =  ds.sel(hour=slice(6, 18))
                                        
                                    else:
                                        
                                        ds_base = xr.open_mfdataset(f'{rootdir_IFS}/{res}/ML/Processed/{res}_ml_subset_z_2022*.nc',engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_ref)
                                        ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                        
                                        if mode == 'SCu':
                                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                                        else:
                                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                                        
                                        sub_mean = sub.groupby('time.hour').mean(dim='time')
                                        sub_std = sub.groupby('time.hour').std(dim='time')
                                        
                                        #Rename
                                        sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                                        sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                                        
                                        #ds
                                        ds = xr.merge([sub_mean, sub_std])
                                        ds = ds[sorted(ds.data_vars)]
                                        
                                        #Selection of hours
                                        ds =  ds.sel(hour=slice(6, 18))
                                        
                                y_IFS = ds[f'{varname}_mean'].values*multiplier
                                y_IFS_std = ds[f'{varname}_std'].values

                                ax.plot(x_IFS, y_IFS, marker='o', markersize=4, markeredgecolor='k', markeredgewidth=0.5,color=IFS_colour)
                                
                                if plot_IFS == True:
                                    if var not in ['hcc','mcc','lcc']:
                                        
                                        if var in vars_tower or var=='h' or var in vars_rad:
                                            plot_label = f'IFS {res} ({key})'
                                        else:
                                            if var == 'CO2': 
                                                plot_label = f'IFS {res} {z_sel_CO2} m ({key})'
                                            else:
                                                plot_label = f'IFS {res} {z_ref} m ({key})'

                                        ax.plot([],[], color=IFS_colour, marker='o', linestyle='-', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0,label=plot_label)
                                    else:
                                        ax.plot([],[], color=IFS_colour, marker='', linestyle='-', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0,label=f'IFS {res} ({key})')
                                    
                                    
                                    if IFS_std == True:
                                        ax.errorbar(x_IFS, y_IFS, yerr=y_IFS_std,
                                                    linestyle='', marker='o', markersize=4,
                                                    markeredgecolor='k', markeredgewidth=0.5,
                                                    color=IFS_colour,
                                                    ecolor='k', elinewidth=2.5, alpha=1.0)

                                        ax.errorbar(x_IFS, y_IFS, yerr=y_IFS_std,
                                                    linestyle='', marker='o', markersize=4,
                                                    markeredgecolor='k', markeredgewidth=0.5,
                                                    color=IFS_colour,
                                                    ecolor=IFS_colour, elinewidth=1.0, alpha=1.0)
                                
                            elif mode == 'daily':
                             
                                if datatype == 'sfc':
                                    ds = xr.open_mfdataset(f"{rootdir_IFS}/{res}/SFC/Processed/{res}_sfc_subset_km_Amazon_2022*.nc",engine='h5netcdf').mean(dim=['latitude','longitude'])
                                else:
                                    if var == 'CO2':
                                        z_sel_CO2 = heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]
                                        ds = xr.open_mfdataset(f'{rootdir_IFS}/{res}/ML/Processed/{res}_ml_subset_z_ghg_2022*.nc',engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_sel_CO2)
                                    else:
                                        ds = xr.open_mfdataset(f'{rootdir_IFS}/{res}/ML/Processed/{res}_ml_subset_z_2022*.nc',engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_ref)
                                ds['time'] = pd.to_datetime(ds['time'].values) - pd.Timedelta(hours=4)  # Adjust for UTC-4
                                
                                ds = ds.sel(time=slice('2022-08-09 06:00:00', '2022-08-18 18:00:00'))
                                        
                                x_IFS = ds.time.values
                                y_IFS = ds[varname].values
                                
                                indices = [list(x_IFS).index(item) for item in list(x_IFS) if item in list(x_OBS)]
                                
                                y_IFS = y_IFS[indices]*multiplier
                                x_IFS = x_IFS[indices]
                                
                                #Add a nan value each day at 19:00 LT 
                                for i in range(len(x_IFS)):
                                    
                                    datetimer = pd.to_datetime(x_IFS[i])
                                    
                                    if datetimer.hour == 18:
                                        y_IFS = np.insert(y_IFS, i+1, np.nan)
                                        x_IFS = np.insert(x_IFS, i+1, datetime(datetimer.year, datetimer.month, datetimer.day, 19, 0, 0))
                                
                                if plot_IFS == True:
                                    if var not in ['hcc','mcc','lcc']:
                                        if var in vars_tower or var=='h' or var in vars_rad:
                                            ax.plot(x_IFS, y_IFS[:], color=IFS_colour, label=f'IFS {res} ({key})',marker='o', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0)
                                        else:
                                            if var == 'CO2':
                                                ax.plot(x_IFS, y_IFS[:], color=IFS_colour, label=f'IFS {res} {z_sel_CO2} m ({key})',marker='o', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0)
                                            else:
                                                ax.plot(x_IFS, y_IFS[:], color=IFS_colour, label=f'IFS {res} {z_ref} m ({key})',marker='o', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0)
                                    else:
                                        ax.plot(x_IFS, y_IFS[:], color=IFS_colour, label=f'IFS {res} ({key})',marker='', markersize=4, markeredgecolor='k', markeredgewidth=0.5, alpha=1.0,linewidth=1)

                            ds.close()
                            IFS_data_y.append(y_IFS)
                                   
                    #--------------------------------------------                
                    #Detailing
                    ax.set_ylabel(label)
                    
                    ax.tick_params(axis='x')
                    ax.tick_params(axis='y')

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    ax.yaxis.set_minor_locator(AutoMinorLocator())
                    
                    ax.tick_params(which='major')
                    ax.tick_params(which='minor')
                    
                    #x-axis
                    if mode != 'daily':
                        ax.set_xlabel('LT (hour)')
                        ax.set_xticks(np.arange(6, 19, 2),np.round(np.arange(6, 19, 2),1))
                        ax.xaxis.set_minor_locator(AutoMinorLocator())

                    elif mode == 'daily':
                        
                        xticks_range = pd.date_range(start="2022-08-09 00:00:00", end="2022-08-19 00:00:00", freq='6h')
                        xticks_range = [datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') for x in xticks_range]
                        ax.set_xticks(xticks_range)
                        ax.set_xticklabels(np.array([x.strftime('%H') for x in xticks_range]).astype(int))

                        ax.set_xlim(datetime.strptime('2022-08-09 00:00:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2022-08-19 00:00:00', '%Y-%m-%d %H:%M:%S'))

                        #Shading for Shallow Convective and Shallow-to-Deep Convective Days
                        X = 0.08
                        SCu_list = ['2022-08-09', '2022-08-10', '2022-08-11', '2022-08-15', '2022-08-17', '2022-08-18']
                        DeepCu_list = ['2022-08-12', '2022-08-13', '2022-08-14', '2022-08-16']

                        # Loop over full date range for vertical lines and shading
                        for day in pd.date_range('2022-08-09', '2022-08-18'):
                            day_str = day.strftime('%Y-%m-%d')
                            start_dt = datetime.strptime(day_str, '%Y-%m-%d')
                            end_dt = start_dt + timedelta(days=1)

                            # Add vertical line at midnight (starting from 10th)
                            if start_dt >= datetime(2022, 8, 10):
                                ax.axvline(start_dt, color='tab:gray', linestyle='--', linewidth=axiswidth*2)

                            # Add shading
                            if day_str in SCu_list:
                                ax.axvspan(start_dt, end_dt, color='darkred', alpha=X)
                            elif day_str in DeepCu_list:
                                ax.axvspan(start_dt, end_dt, color='tab:blue', alpha=X)


                        for day in range(9, 19):  # From 09 to 18 August
                            date_str = f'2022-08-{day:02d} 12:00:00'
                            display_str = f'{day:02d}-08'
                            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

                            ax.text(dt, 1.03, display_str,
                                    ha='center', va='center',
                                    transform=ax.get_xaxis_transform())

                    #y-axis
                    if var == 'EF':
                        ax.set_ylim(0,1.2)
                        ax.set_yticks(np.arange(0, 1.4, 0.2),np.round(np.arange(0, 1.4, 0.2),1))
                    elif var == 'WD':
                        ax.set_ylim(0,360)
                        ax.set_yticks(np.arange(0, 361, 90),np.round(np.arange(0, 361, 90),1))
                    elif var == 'clouds':
                        ax.set_ylim(0,6000)
                    elif var == 'cloud cover':
                        ax.set_yticks(np.arange(0, 1.4, 0.2),np.round(np.arange(0, 1.4, 0.2),1))
                        ax.set_ylim(0,1)
                    elif var == 'CO2' and z_ref == 298:
                        ax.set_ylim(400,450)
                        ax.set_yticks(np.arange(400, 451, 10),np.round(np.arange(400, 451, 10),1))
                    elif var == 'CO2' and z_ref == 50: 
                        ax.set_ylim(400,500)
                        ax.set_yticks(np.arange(400, 520, 20),np.round(np.arange(400, 520, 20),1))
                    elif var == 'LWnet' and mode == 'DeepCu':
                        ax.set_ylim(-100,0)
                        ax.set_yticks(np.arange(-100, 1, 20),np.round(np.arange(-100, 1, 20),1).astype(int))
                    else:
                        ymin, ymax = ax.get_ylim()
                        ymin = step * np.round(ymin / step)
                        ymax = step * np.round(ymax / step)
                        ymax = ymax + step
                        ymin = ymin

                        ax.set_yticks(np.arange(np.floor(ymin / step) * step, np.ceil(ymax / step) * step + step, step))

                        ax.set_ylim(ymin,ymax)    
                                    
                    ax.axhline(0, color='tab:gray', linewidth=1.5, ls='--')
                        
                    #Add grid
                    ax.grid(linestyle=':',color='tab:gray',alpha=0.3,linewidth=axiswidth*2)

                    #Add legend
                    if mode !='daily':
                        legend = ax.legend(loc='lower left', bbox_to_anchor= (-0.02, 1.05), ncol=2,
                                    borderaxespad=0, frameon=False,markerscale=1,borderpad=0.7)
                    else:
                        legend = ax.legend(loc='lower left', bbox_to_anchor= (-0.02, 1.05), ncol=3,
                            borderaxespad=0, frameon=False,markerscale=1,borderpad=0.7)

                    for lh in legend.get_patches() + legend.get_lines():
                                    lh.set_alpha(1)

                    #Export
                    if not os.path.exists(f"{outdir}/{var}/"):
                        os.makedirs(f"{outdir}/{var}/")

                    if var not in ['h','mass_flux','hcc','mcc','lcc']:
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_{z_ref}m.pdf",bbox_inches='tight')   
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_{z_ref}m.jpg",bbox_inches='tight',dpi=300)
                        #plt.savefig(f"{outdir}/{var}/{var}_{mode}_{z_ref}m.svg",bbox_inches='tight')
                    else:
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}.pdf",bbox_inches='tight')   
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}.jpg",bbox_inches='tight',dpi=300)
                        #plt.savefig(f"{outdir}/{var}/{var}_{mode}.svg",bbox_inches='tight')
                    
                    #plt.show()
                    plt.close(fig)
            
                #-------------------------  S T A S T I C S  -------------------------
                
                #Align all data to IFS
                
                if not (var == 'mass flux' and mode in ['DeepCu', 'daily']):
                
                    #OBS
                    if var == 'mass flux':
                        IFS_data_y = [x_IFS]
                    
                    if var not in ['hcc','mcc','lcc','mass flux']:
                        x_IFS_c, y_IFS_c, y_OBS_c = testbed_functions.align_data_to_reference(x_IFS,x_OBS,IFS_data_y[0],y_OBS)

                    #DALES
                    if mode == 'SCu': 
                        x_IFS_c, y_IFS_c, y_DALES_c = testbed_functions.align_data_to_reference(x_IFS,x_DALES,IFS_data_y[0],y_DALES)
                        x_IFS_c, y_IFS_no_clouds_c, y_DALES_no_clouds_c = testbed_functions.align_data_to_reference(x_IFS,x_DALES,IFS_data_y[0],y_DALES_no_clouds)
                            
                    #CLASS
                    if mode == 'SCu': 
                        x_IFS_c, y_IFS_c, y_CLASS_c = testbed_functions.align_data_to_reference(x_IFS,x_CLASS,IFS_data_y[0],y_CLASS)
                        x_IFS_c, y_IFS_no_clouds_c, y_CLASS_no_clouds_c = testbed_functions.align_data_to_reference(x_IFS,x_CLASS,IFS_data_y[0],y_CLASS_no_clouds)
                    
                
                    if mode == 'SCu':
                        if var in ['hcc','mcc','lcc']:
                            data = {'CLASS CLOUDS' : y_CLASS_c,
                                    'CLASS NO CLOUDS' : y_CLASS_no_clouds_c,
                                    'DALES CLOUDS' : y_DALES_c,
                                    'DALES NO CLOUDS' : y_DALES_no_clouds_c,
                                    'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}
                        
                        elif var == 'mass flux':
                            data = {'CLASS CLOUDS' : y_CLASS_c,
                                    'CLASS NO CLOUDS' : y_CLASS_no_clouds_c,
                                    'DALES CLOUDS' : y_DALES_c,
                                    'DALES NO CLOUDS' : y_DALES_no_clouds_c}
                        
                        else:
                            data = {'obs' : y_OBS_c,
                                    'CLASS CLOUDS' : y_CLASS_c,
                                    'CLASS NO CLOUDS' : y_CLASS_no_clouds_c,
                                    'DALES CLOUDS' : y_DALES_c,
                                    'DALES NO CLOUDS' : y_DALES_no_clouds_c,
                                    'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}
                    
                    elif mode == 'DeepCu':
                    
                        if var in ['hcc','mcc','lcc']:
                            data = {'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}
                                                
                        elif var!='mass flux':
                            data = {'obs' : y_OBS_c,
                                    'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}
                    
                    elif mode == 'daily':
                        
                        if var in ['hcc','mcc','lcc']:
                            data = {'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}
                        else:
                            data = {'obs' : y_OBS_c,
                                    'IFS ia61 (28km)' : IFS_data_y[0],
                                    'IFS ia62 (9km)' : IFS_data_y[1],
                                    'IFS ia63 (4.4km)' : IFS_data_y[2]}

                    df = pd.DataFrame(data)

                    metrics_list = []

                    for var1, var2 in combinations(df.columns, 2):
                        corr = df[var1].corr(df[var2])
                        rms = testbed_functions.calc_rmse(df[var1], df[var2])
                        b = testbed_functions.calc_bias(df[var1], df[var2])
                        am = abs(testbed_functions.calc_amplitude(df[var1]) - testbed_functions.calc_amplitude(df[var2]))
                        
                        metrics_list.append({'var1': var1, 'var2': var2, 
                                            'Correlation': corr, 'RMSE': rms, 'Bias': b, 'DXR': am})

                    results_df = pd.DataFrame(metrics_list)

                    metrics = ['Correlation', 'RMSE', 'Bias', 'DXR']
                    
                    #---------------------------------- V I S U A L I S E ------------------------------------------------
                    
                    fig, axes = plt.subplots(2, 2, figsize=(16*1.2, 12*1.2),dpi=300)

                    for ax, metric in zip(axes.flatten(), metrics):
                        
                        # Create pivot table
                        pivot = results_df.pivot(index='var1', columns='var2', values=metric)
                        pivot = pivot.reindex(index=df.columns, columns=df.columns)
                        
                        # Mirror the lower triangle to make a full matrix
                        full_matrix = pivot.copy()
                        full_matrix = full_matrix.combine_first(pivot.T)    
                        
                        if metric == 'Correlation':
                            cmap = 'coolwarm_r'
                            center = None
                        elif metric == 'Bias':
                            from matplotlib.colors import LinearSegmentedColormap

                            original = plt.get_cmap('coolwarm')
                            colors = original(np.linspace(0, 1, 256))

                            # Fade the middle part into white over a few indices around the center
                            fade = 5  # Number of color entries around the center to fade into white
                            center_index = len(colors) // 2

                            for i in range(-fade, fade + 1):
                                idx = center_index + i
                                if 0 <= idx < len(colors):
                                    # Linear fade to white based on distance from center
                                    weight = abs(i) / fade
                                    colors[idx][:3] = weight * colors[idx][:3] + (1 - weight) * np.array([1, 1, 1])

                            # Create the new colormap
                            cmap = LinearSegmentedColormap.from_list('custom_coolwarm_white_center', colors)
                            
                            center = 0
                        else:
                            cmap = 'coolwarm'
                            center = None
                            
                        heatmap = sns.heatmap(full_matrix, annot=True, fmt=".2f", cmap=cmap, ax=ax,linewidths=1,linecolor='k',center=center)
                        

                        cbar = heatmap.collections[0].colorbar

                        locator = MaxNLocator(nbins=6) 
                        cbar.locator = locator
                        cbar.update_ticks()

                        #Detailing
                        
                        #invert yaxis
                        ax.invert_yaxis()
                        
                        if metric != 'DXR':
                            ax.set_title(rf'$\mathbf{{{metric}}}$')
                        else:
                            ax.set_title(rf'$\mathbf{{|\: Difference \: {metric} \:|}}$')
                        
                        ax.set_xlabel(r'$\mathbf{Reference:}$')
                        ax.tick_params(axis='x',pad=30)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        ax.xaxis.set_label_coords(0.5,-0.05)

                        ax.set_ylabel(r'$\mathbf{Forecasted:}$',rotation=0)
                        ax.yaxis.set_label_coords(-0.15, 1.05)
                        ax.tick_params(axis='y')

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        ax.tick_params(which='major')
                        ax.tick_params(which='minor')
                        

                    text_ref = 'Shallow Convective Regime (ShCu)' if mode == 'SCu' else 'Shallow-to-Deep Convective Regime (ShDeep)' if mode == 'DeepCu' else 'Daily'

                    if var not in ['h','mass_flux','hcc','mcc','lcc']:
                        if var == 'CO2':
                            z_sel_CO2 = heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]
                            fig.suptitle(f'{label} | {text_ref} | Statistics | z = {str(z_sel_CO2)} m',fontsize=plt.rcParams['font.size']*2, y=1.05)
                        else:
                            fig.suptitle(f'{label} | {text_ref} | Statistics | z = {str(z_ref)} m',fontsize=plt.rcParams['font.size']*2, y=1.05)
                    else:
                        fig.suptitle(f'{label} | {text_ref} | Statistics',fontsize=plt.rcParams['font.size']*2, y=1.05)

                    plt.tight_layout()
                    
                    if not os.path.exists(f"{outdir}/{var}/"):
                        os.makedirs(f"{outdir}/{var}/")

                    #Export
                    if var not in ['h','mass_flux','hcc','mcc','lcc']:
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics_{z_ref}m.pdf",bbox_inches='tight')
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics_{z_ref}m.jpg",bbox_inches='tight',dpi=300)
                        #plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics_{z_ref}m.svg",bbox_inches='tight')
                    else:
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics.pdf",bbox_inches='tight')   
                        plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics.jpg",bbox_inches='tight',dpi=300)
                        #plt.savefig(f"{outdir}/{var}/{var}_{mode}_statistics.svg",bbox_inches='tight')

                    #plt.show()
                    plt.close()
                    

#%%
#-----------------------------------------------------------------------------------------------------------
#                                   V E R T I C A L    P R O F I L E S 
#-----------------------------------------------------------------------------------------------------------


vertical_vars = {
    
    'theta' : {'label' : r'$\theta$ (K)',
            'step': 2,
            'varname': 'theta',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'q' : {'label' : 'q (kg kg$^{-1}$)',
            'step': 2,
            'varname': 'q',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'U': {'label' : 'U (m s$^{-1}$)',
            'step': 1,
            'varname': 'ws',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'WD': {'label' : 'WD (°)',
            'step': 90,
            'varname': 'wdir',
            'datatype' : 'ml',
            'multiplier': 1},
    

    'CO2': {'label' : 'CO$_2$ (ppm)',
            'step': 5,
            'varname': 'co2',
            'datatype' : 'ml',
            'multiplier': 1},   
    
}

modes = ['DeepCu','SCu']


plot_CLASS = False

if plot_profiles == True:
    for var in vertical_vars.keys():
        
        print(var)
 
        #x_ref
        if var != 'CO2':
            x_ref = [6,9,12,15,18] 
        else:
            x_ref = [6,9,12,13,15,18]

        #----------------------------------------- L O A D    D A T A ------------------------------------------------
        
        #----------------------------------------- O B S E R V A T I O N S -------------------------------------------
        
        #z
        z_OBS = locals()[f'SCu_values_z_ref'][0][0]    

        #y
        if var!= 'CO2':
            if var == 'U':
                var_s = 'wind_speed'
            elif var == 'WD':
                var_s = 'wind_dir'
            else:
                var_s = var

            y_SCu_obs = locals()[f'SCu_values_{var_s}_ref']
            y_SCu_obs_mean = np.nanmean(y_SCu_obs,axis=0)
            y_SCu_obs_std = np.nanstd(y_SCu_obs,axis=0)

            y_DeepCu_obs = locals()[f'DeepCu_values_{var_s}_ref']
            y_DeepCu_obs_mean = np.nanmean(y_DeepCu_obs,axis=0)
            y_DeepCu_obs_std = np.nanstd(y_DeepCu_obs,axis=0)

        else:
            x_OBS_9LT = flight1_data['co2']
            z_OBS_9LT = flight1_data['altitude']

            x_OBS_13LT = flight2_data['co2']
            z_OBS_13LT = flight2_data['altitude']

        #OBS - h
        y_SCu_h = np.nanmean(np.array(SCu_values_h),axis=0)
        y_SCu_h[0] = abl_SCu_mean[0]

        if var == 'CO2':    #insert a value at 13 LT
            
            x_OBS = np.array(testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="5min").time)[1:])
            
            y_SCu_h = np.insert(y_SCu_h, 3, abl_SCu_mean[testbed_functions.find_closest(x_OBS, 13)])

        y_DeepCu_h = np.nanmean(np.array(DeepCu_values_h),axis=0)
        y_DeepCu_h[0] = abl_DeepCu_mean[0]

        #------------------------------------------ D A L E S ------------------------------------------------
        
        #Load DALES - x
        x_DALES = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="5min").time)[1:]

        #DALES - y
        y_DALES = []
        y_DALES_no_clouds = []
        for time in x_ref: 
            idx = testbed_functions.find_closest(np.array(x_DALES), time)
            y_DALES.append(DALES_profiles[var][idx,:])
            y_DALES_no_clouds.append(DALES_profiles_no_clouds[var][idx,:])


        #----------------------------------------- V I S U A L I S E ------------------------------------------------

        label = vertical_vars[var]['label']
        step = vertical_vars[var]['step']
        varname = vertical_vars[var]['varname']
        datatype = vertical_vars[var]['datatype']
        multiplier = vertical_vars[var]['multiplier']
        
        for mode in modes:
            
            print(mode)
            
            if var != 'CO2':
                fig, ax = plt.subplots(1,5,figsize=(11.69,8.27/2),dpi=300,sharey=True)
                ax1,ax2,ax3,ax4,ax5 = ax.flatten()
            elif var == 'CO2' and mode == 'SCu':
                fig, ax = plt.subplots(1,6,figsize=(11.69,8.27/2),dpi=300,sharey=True)
                ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
            
            IFS_data_y = []
            
            for i in range(len(x_ref)):
                
                #OBS
                #Visualise data for ShCu and ShDeep
                if var != 'CO2': 

                    #Select variable
                    y_OBS = locals()[f'y_{mode}_obs_mean']
                    y_OBS_std = locals()[f'y_{mode}_obs_std']
                    
                    #Define colour for OBS
                    colour = 'tab:red' if mode == 'SCu' else 'tab:blue'

                    idx_OBS = testbed_functions.find_closest(np.array(z_OBS), 3500)
                    locals()[f'ax{i+1}'].plot(y_OBS[i,:idx_OBS],z_OBS[:idx_OBS],linewidth=6.5, color=colour, ls="", marker='o', markersize=1, markeredgecolor='k', markeredgewidth=0.5, alpha=0.2,label='OBS ShCu' if mode == 'SCu' else 'OBS ShDeep')
                    locals()[f'ax{i+1}'].fill_betweenx(z_OBS[:idx_OBS],y_OBS[i,:idx_OBS]-y_OBS_std[i,:idx_OBS],y_OBS[i,:idx_OBS]+y_OBS_std[i,:idx_OBS],color=colour,alpha=0.15)
                
                #OBS at 9 and 13 LT for CO2
                elif var == 'CO2' and mode == 'SCu': 
                    locals()[f'ax{i+1}'].plot([],[],linewidth=6.5, color='tab:red', ls="", marker='o', markersize=1, markeredgecolor='k', markeredgewidth=0.5, alpha=0.2,label='OBS ShCu' if mode == 'SCu' else 'OBS ShDeep')    
                    if x_ref[i] == 9:
                        locals()[f'ax{i+1}'].plot(x_OBS_9LT,z_OBS_9LT,linewidth=6.5, color='tab:red', ls="", marker='o', markersize=1, markeredgecolor='k', markeredgewidth=0.5, alpha=0.2)
                    elif x_ref[i] == 13:
                        locals()[f'ax{i+1}'].plot(x_OBS_13LT,z_OBS_13LT,linewidth=6.5, color='tab:red', ls="", marker='o', markersize=1, markeredgecolor='k', markeredgewidth=0.5, alpha=0.2)
                
                #CLASS
               
                #Only visualise CLASS for ShCu
                if var != 'U' and var != 'WD' and mode == 'SCu': 
                    
                    # #Construct vertical profile following mixed-layer theory (to use varying r_c)
                    # if var!= 'CO2':
                    #     index_h = testbed_functions.find_closest(np.array(z_OBS), locals()[f'y_{mode}_h'][i])
                    #     index_h_above = index_h + 1
                        
                    #     index_top = testbed_functions.find_closest(np.array(z_OBS), 3500)
                        
                    #     delta_y = z_OBS[index_top] - z_OBS[index_h_above]
                    #     delta_x = y_OBS[i][index_top] - y_OBS[i][index_h_above]
                        
                    #     #Lapse rate
                    #     r_c = delta_x/delta_y   
                        
                    # else:
                    #     index_h = testbed_functions.find_closest(np.array(z_DALES), locals()[f'y_{mode}_h'][i])
                    #     index_h_above = index_h + 1
                    #     index_top = testbed_functions.find_closest(np.array(z_DALES), 3500)
                    #     delta_y = z_DALES[index_top] - z_DALES[index_h_above]
                    #     delta_x = y_DALES[i][index_top] - y_DALES[i][index_h_above]
                        
                    #     #Lapse rate
                    #     r_c = delta_x/delta_y   
                                
                    #CLOUDS
                    proftime = np.where(SCu_run.out.t==x_ref[i]+4)[0][0]

                    Ct_h = SCu_run.out.h[proftime]
                    if var == 'theta':
                        Ct_thetaprof = SCu_run.out.theta[proftime]
                        Ct_dthetaprof = SCu_run.out.dtheta[proftime]
                        Ct_gammathetaprof = SCu_run.input.gammatheta  
                    elif var == 'q':
                        Ct_thetaprof = SCu_run.out.q[proftime] * 1000
                        Ct_dthetaprof = SCu_run.out.dq[proftime] * 1000
                        Ct_gammathetaprof = SCu_run.input.gammaq * 1000
                    elif var == 'CO2':
                        Ct_thetaprof = SCu_run.out.CO2[proftime]
                        Ct_dthetaprof = SCu_run.out.dCO2[proftime]
                        Ct_gammathetaprof = SCu_run.input.gammaCO2

                    C_thetaprof = np.zeros(3500)
                    C_height = np.arange(0,3500)
                    
                    hindex = np.where(C_height>Ct_h)[0][0]
                    C_thetaprof[:hindex] = Ct_thetaprof
                    C_thetaprof[hindex] = Ct_thetaprof + Ct_dthetaprof
                    C_thetaprof[hindex+1:] = C_thetaprof[hindex] + Ct_gammathetaprof * (C_height[hindex+1:]-C_height[hindex])
                    
                    C_height_ad = C_height[::10]
                    C_height_ad = C_height_ad[1:]
                    C_height_ad = C_height_ad[::2]
                    
                    C_thetaprof_ad = C_thetaprof[::10]
                    C_thetaprof_ad = C_thetaprof_ad[1:]
                    C_thetaprof_ad = C_thetaprof_ad[::2]
                    
                    if plot_CLASS == True:
                        locals()[f'ax{i+1}'].plot(C_thetaprof,C_height,color='tab:blue',label='CLASS CLOUDS')   
                    
                    #NO CLOUDS
                    Ct_h = SCu_run_no_clouds.out.h[proftime]
                    if var == 'theta':
                        Ct_thetaprof = SCu_run.out.theta[proftime]
                        Ct_dthetaprof = SCu_run.out.dtheta[proftime]
                        Ct_gammathetaprof = SCu_run.input.gammatheta  
                    elif var == 'q':
                        Ct_thetaprof = SCu_run.out.q[proftime] * 1000
                        Ct_dthetaprof = SCu_run.out.dq[proftime] * 1000
                        Ct_gammathetaprof = SCu_run.input.gammaq * 1000
                    elif var == 'CO2':
                        Ct_thetaprof = SCu_run.out.CO2[proftime]
                        Ct_dthetaprof = SCu_run.out.dCO2[proftime]
                        Ct_gammathetaprof = SCu_run.input.gammaCO2

                    C_thetaprof = np.zeros(3500)
                    C_height = np.arange(0,3500)
                    
                    hindex = np.where(C_height>Ct_h)[0][0]
                    C_thetaprof[:hindex] = Ct_thetaprof
                    C_thetaprof[hindex] = Ct_thetaprof + Ct_dthetaprof
                    C_thetaprof[hindex+1:] = C_thetaprof[hindex] + Ct_gammathetaprof * (C_height[hindex+1:]-C_height[hindex])
                    
                    C_height_ad = C_height[::10]
                    C_height_ad = C_height_ad[1:]
                    C_height_ad = C_height_ad[::2]
                    
                    C_thetaprof_ad = C_thetaprof[::10]
                    C_thetaprof_ad = C_thetaprof_ad[1:]
                    C_thetaprof_ad = C_thetaprof_ad[::2]
                    
                    if plot_CLASS == True:
                        locals()[f'ax{i+1}'].plot(C_thetaprof,C_height,color='tab:blue',linestyle='--',label='CLASS CLOUDS')  
                    
                #DALES
                idx_DALES = testbed_functions.find_closest(np.array(z_DALES), 3500)
                if mode == 'SCu' and plot_DALES==True:
                    locals()[f'ax{i+1}'].plot(y_DALES[i][:idx_DALES],z_DALES[:idx_DALES],color='tab:green', ls="-", label='DALES CLOUDS')
                    locals()[f'ax{i+1}'].plot(y_DALES_no_clouds[i][:idx_DALES],z_DALES[:idx_DALES],color='tab:green', ls="--", label='DALES NO CLOUDS')
                    
                #IFS
                y_IFS_data = []
                for key in resolutions.keys():
                    
                    res = resolutions[key][0]
                    IFS_colour = resolutions[key][1]

                    if var != 'CO2':
                        ds_base = xr.open_mfdataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_ml_subset_z_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude'])
                        ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                    
                        if mode == 'SCu':
                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                        else:
                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                        
                        sub_mean = sub.groupby('time.hour').mean(dim='time')
                        sub_std = sub.groupby('time.hour').std(dim='time')
                        
                        #Rename
                        sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                        sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                        
                        #ds
                        ds = xr.merge([sub_mean, sub_std])
                        ds = ds[sorted(ds.data_vars)]
                        
                        #Selection of hours
                        ds =  ds.sel(hour=x_ref[i])
                    else:
                        ds_base = xr.open_mfdataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_ml_subset_z_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude'])
                        ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                        
                        if mode == 'SCu':
                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                        else:
                            sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                        
                        sub_mean = sub.groupby('time.hour').mean(dim='time')
                        sub_std = sub.groupby('time.hour').std(dim='time')
                        
                        #Rename
                        sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                        sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                        
                        #ds
                        ds = xr.merge([sub_mean, sub_std])
                        ds = ds[sorted(ds.data_vars)]
                        
                        #Selection of hours
                        ds =  ds.sel(hour=x_ref[i])

                    x_IFS = ds[f'{varname}_mean'].values*multiplier
                    x_IFS_std = ds[f'{varname}_std'].values
                    z_IFS = ds['z'].values
                    idx_IFS = testbed_functions.find_closest(z_IFS, 3500)

                    if plot_IFS == True:
                        
                        if var != 'CO2':
                            locals()[f'ax{i+1}'].plot(x_IFS[:idx_IFS+1],z_IFS[:idx_IFS+1], color=IFS_colour, label=f'IFS {res} ({key})')

                            if IFS_std == True:
                                locals()[f'ax{i+1}'].fill_betweenx(z_IFS[:idx_IFS+1], x_IFS[:idx_IFS+1]-x_IFS_std[:idx_IFS+1], x_IFS[:idx_IFS+1]+x_IFS_std[:idx_IFS+1], color=IFS_colour, alpha=0.15)
                        else:
                            locals()[f'ax{i+1}'].plot(x_IFS[:idx_IFS+1],z_IFS[:idx_IFS+1], color=IFS_colour, label=f'IFS {res} ({key})')
                            
                            if IFS_std == True:
                                locals()[f'ax{i+1}'].fill_betweenx(z_IFS[:idx_IFS+1], x_IFS[:idx_IFS+1]-x_IFS_std[:idx_IFS+1], x_IFS[:idx_IFS+1]+x_IFS_std[:idx_IFS+1], color=IFS_colour, alpha=0.15)
                    ds.close()
                    
                    y_IFS_data.append(x_IFS[:idx_IFS+1])
                
                IFS_data_y.append(y_IFS_data) 
           
                #Limits
                locals()[f'ax{i+1}'].set_title(f'{x_ref[i]} LT',fontsize=plt.rcParams['font.size'],y=0.95)
                locals()[f'ax{i+1}'].set_ylim(0,3500)
                locals()[f'ax{i+1}'].set_yticks(np.arange(0, 3501, 500),np.round(np.arange(0, 3501, 500),1))

                if var == 'WD':
                    locals()[f'ax{i+1}'].set_xlim(0, 360)
                    locals()[f'ax{i+1}'].set_xticks(np.arange(0, 361, 90))
                    locals()[f'ax{i+1}'].set_xticklabels(np.arange(0, 361, 90).astype(int))
                    locals()[f'ax{i+1}'].xaxis.set_minor_locator(MultipleLocator(45))
                
                elif var == 'CO2':
                
                    desired_ticks = 3
                    xmin_raw, xmax_raw = locals()[f'ax{i+1}'].get_xlim()
                    xmin = testbed_functions.round_if_close(xmin_raw)
                    xmax = testbed_functions.round_if_close(xmax_raw)

                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if xmin == xmax:
                        xmax += 1
                    
                    if x_ref[i] == 15:
                        xmax = xmax+5

                    ticks = np.linspace(xmin, xmax, desired_ticks)
                    major_step = np.diff(ticks).mean()

                    locals()[f'ax{i+1}'].set_xticks(ticks)
                    locals()[f'ax{i+1}'].set_xticklabels([f"{t:.0f}" for t in ticks])
                    locals()[f'ax{i+1}'].set_xlim(xmin - major_step / 2, xmax)
                    locals()[f'ax{i+1}'].xaxis.set_minor_locator(MultipleLocator(major_step / 2))
                    
                    if x_ref[i] == 15:
                        locals()[f'ax{i+1}'].set_xlim(xmin - major_step / 2, 425)
                 
                                        
                else:
                    desired_ticks = 4
                    xmin_raw, xmax_raw = locals()[f'ax{i+1}'].get_xlim()
                    if var == 'theta':
                        xmin = testbed_functions.round_if_close(xmin_raw-2)
                    else:
                        xmin = testbed_functions.round_if_close(xmin_raw)
                    xmax = testbed_functions.round_if_close(xmax_raw)

                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if xmin == xmax:

                        xmax += 1

                    ticks = np.linspace(xmin, xmax, desired_ticks)
                    major_step = np.diff(ticks).mean()

                    locals()[f'ax{i+1}'].set_xticks(ticks)
                    locals()[f'ax{i+1}'].set_xticklabels([f"{t:.0f}" for t in ticks])
                    locals()[f'ax{i+1}'].set_xlim(xmin, xmax)
                    locals()[f'ax{i+1}'].xaxis.set_minor_locator(MultipleLocator(major_step / 2))
                            
                ax1.set_ylabel('z (m)')
                
                locals()[f'ax{i+1}'].set_xlabel(label)

                locals()[f'ax{i+1}'].tick_params(axis='x')
                locals()[f'ax{i+1}'].tick_params(axis='y')

                locals()[f'ax{i+1}'].spines['top'].set_visible(False)
                locals()[f'ax{i+1}'].spines['right'].set_visible(False)

                locals()[f'ax{i+1}'].yaxis.set_minor_locator(AutoMinorLocator())
                locals()[f'ax{i+1}'].xaxis.set_minor_locator(AutoMinorLocator())
                locals()[f'ax{i+1}'].tick_params(which='major')
                locals()[f'ax{i+1}'].tick_params(which='minor')
                locals()[f'ax{i+1}'].grid(linestyle=':',color='tab:gray',alpha=0.3,linewidth=axiswidth*2)

            plt.subplots_adjust(wspace=0.4, hspace=0.6)

            legend = ax1.legend(loc='lower left', bbox_to_anchor= (-0.3, 1.2), ncol=4,
                                borderaxespad=0, frameon=False,markerscale=7,borderpad=0.7)

            for lh in legend.get_patches() + legend.get_lines():
                            lh.set_alpha(1)
            
            #Export
            if not os.path.exists(f"{outdir}/{var}/"):
                os.makedirs(f"{outdir}/{var}/")
            
            plt.savefig(f"{outdir}/{var}/{var}_{mode}_vertical_profiles.pdf",bbox_inches='tight')
            plt.savefig(f"{outdir}/{var}/{var}_{mode}_vertical_profiles.jpg",bbox_inches='tight',dpi=300)
            #plt.savefig(f"{outdir}/{var}/{var}_{mode}_vertical_profiles.svg",bbox_inches='tight')
            
            #plt.show()
            plt.close()
            
   
            #--------------------------- S T A T I S T I C S ---------------------------


            if mode == 'DeepCu' and var =='CO2':
                pass
            else:
                statistics = []
                
                for i in range(len(x_ref)):

                    
                    #OBS
                    if var!= 'CO2':
                        x_aligned, y_dales_aligned, y_obs_aligned = testbed_functions.align_data_to_reference(z_DALES, z_OBS, z_DALES, y_OBS[i])
                    else: 
                        if x_ref[i] == 9:
                            x_aligned, y_dales_aligned, y_obs_aligned = testbed_functions.align_data_to_reference(z_DALES, z_OBS_9LT, z_DALES, x_OBS_9LT)
                        elif x_ref[i] == 13:
                            x_aligned, y_dales_aligned, y_obs_aligned = testbed_functions.align_data_to_reference(z_DALES, z_OBS_13LT, z_DALES, x_OBS_13LT)
                        else:
                            y_obs_aligned = [np.nan] * len(z_DALES)
                    
                    
                    #IFS
                    idx = testbed_functions.find_closest(np.array(z_IFS), 3500)
                    
                    IFS_data_y_aligned = []
                    for j in range(len(IFS_data_y[i])):
                        if var != 'CO2':
                            x_IFS_aligned, y_DALES_aligned, y_IFS_aligned = testbed_functions.align_data_to_reference(z_DALES, z_IFS[:idx+1], z_DALES, IFS_data_y[i][j])
                        else:
                            x_IFS_aligned, y_DALES_aligned, y_IFS_aligned = testbed_functions.align_data_to_reference(z_DALES, z_IFS[:idx+1], z_DALES, IFS_data_y[i][j])
                        IFS_data_y_aligned.append(y_IFS_aligned)
                                        
                    #Combine data
                    if mode == 'SCu':
                        data = {'OBS' : y_obs_aligned,
                                'DALES' : y_DALES[i],
                                'IFS ia61 (28km)' : IFS_data_y_aligned[0],
                                'IFS ia62 (9km)' : IFS_data_y_aligned[1],
                                'IFS ia63 (4.4km)' : IFS_data_y_aligned[2]}

                    elif mode == 'DeepCu' and var!= 'CO2':
                        data = {'OBS' : y_obs_aligned,
                                'IFS ia61 (28km)' : IFS_data_y_aligned[0],
                                'IFS ia62 (9km)' : IFS_data_y_aligned[1],
                                'IFS ia63 (4.4km)' : IFS_data_y_aligned[2]}
                
                    df = pd.DataFrame(data)
                    
                    metrics_list = []
                    for var1, var2 in combinations(df.columns, 2):
                        try:
                            corr = testbed_functions.calc_r(df[var1], df[var2])
                        except:
                            corr = np.nan
                        
                        try:
                            rms = testbed_functions.calc_rmse(df[var1], df[var2])
                        except:
                            rms = np.nan
                        
                        try:
                            b = testbed_functions.calc_bias(df[var1], df[var2])
                        except:
                            b = np.nan
                        
                        metrics_list.append({'var1': var1, 'var2': var2, 
                                            'Correlation': corr, 'RMSE': rms, 'Bias': b})
                    
                    results_df = pd.DataFrame(metrics_list)
                    
                    statistics.append(results_df)
                    
                
                #Average statistics across all time steps
                numeric_dfs = [df[['Correlation', 'RMSE', 'Bias']] for df in statistics]
                df_numeric_avg = pd.concat(numeric_dfs).groupby(level=0).mean(numeric_only=True)

                df_avg = pd.concat([statistics[0][['var1', 'var2']].reset_index(drop=True), df_numeric_avg.reset_index(drop=True)], axis=1)
                
                metrics = ['Correlation', 'RMSE', 'Bias']
                    
                fig, axes = plt.subplots(1, 3, figsize=(18,7),dpi=300) #w,h

                for ax, metric in zip(axes.flatten(), metrics):
                    
                    # Create pivot table
                    pivot = df_avg.pivot(index='var1', columns='var2', values=metric)
                    pivot = pivot.reindex(index=df.columns, columns=df.columns)
                    
                    # Mirror the lower triangle to make a full matrix
                    full_matrix = pivot.copy()
                    full_matrix = full_matrix.combine_first(pivot.T)    
                    
                    if metric == 'Correlation':
                        cmap = 'coolwarm_r'
                        center = None
                    elif metric == 'Bias':
                        from matplotlib.colors import LinearSegmentedColormap

                        original = plt.get_cmap('coolwarm')
                        colors = original(np.linspace(0, 1, 256))

                        # Fade the middle part into white over a few indices around the center
                        fade = 5  # Number of color entries around the center to fade into white
                        center_index = len(colors) // 2

                        for i in range(-fade, fade + 1):
                            idx = center_index + i
                            if 0 <= idx < len(colors):
                                # Linear fade to white based on distance from center
                                weight = abs(i) / fade
                                colors[idx][:3] = weight * colors[idx][:3] + (1 - weight) * np.array([1, 1, 1])

                        # Create the new colormap
                        cmap = LinearSegmentedColormap.from_list('custom_coolwarm_white_center', colors)
                        
                        center = 0
                    else:
                        cmap = 'coolwarm'
                        center = None
                        
                    heatmap = sns.heatmap(full_matrix, annot=True, fmt=".2f", cmap=cmap, ax=ax,linewidths=1,linecolor='k',center=center)
                    

                    cbar = heatmap.collections[0].colorbar

                    locator = MaxNLocator(nbins=6) 
                    cbar.locator = locator
                    cbar.update_ticks()

                    #Detailing
                    
                    #invert yaxis
                    ax.invert_yaxis()
                    
                
                    ax.set_title(rf'$\mathbf{{{metric}}}$')
                    ax.set_xlabel(r'$\mathbf{Reference:}$')
                    ax.tick_params(axis='x',pad=30)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.xaxis.set_label_coords(0.5,-0.05)

                    ax.set_ylabel(r'$\mathbf{Forecasted:}$',rotation=0)
                    ax.yaxis.set_label_coords(-0.15, 1.05)
                    ax.tick_params(axis='y')

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    ax.tick_params(which='major')
                    ax.tick_params(which='minor')


                plt.tight_layout()
                
                ref = 'Shallow Convective Regime (ShCu)' if mode == 'SCu' else 'Deep Convective Regime (DeepCu)'
                
                fig.suptitle(f'{label} | {ref} | Statistics',fontsize=plt.rcParams['font.size'], y=1.05)

           
                #export
                if not os.path.exists(f"{outdir}/{var}/"):
                    os.makedirs(f"{outdir}/{var}/")
                
                fig.savefig(f"{outdir}/{var}/{var}_vertical_{mode}_statistics.pdf",bbox_inches='tight')
                fig.savefig(f"{outdir}/{var}/{var}_vertical_{mode}_statistics.jpg",bbox_inches='tight',dpi=300)
                #fig.savefig(f"{outdir}/{var}/{var}_vertical_{mode}_statistics.svg",bbox_inches='tight')
                
                #plt.show()
                plt.close()
        


#%%
#-----------------------------------------------------------------------------------------------------------
#                                             M E T R I C S 
#-----------------------------------------------------------------------------------------------------------

#------------------------------------------ D X R ----------------------------------------------------------

#DXR = x_max - x_min

variables = {
    
    'theta' : {'label' : r'$\theta$ (K)',
            'step': 2,
            'varname': 'theta',
            'datatype' : 'ml',
            'multiplier': 1},
    
    'q' : {'label' : 'q (kg kg$^{-1}$)',
            'step': 2,
            'varname': 'q',
            'datatype' : 'ml',
            'multiplier': 1},

    'CO2': {'label' : 'CO$_2$ (ppm)',
            'step': 20,
            'varname': 'co2',
            'datatype' : 'ml',
            'multiplier': 1},
    
}

modes = ['SCu', 'DeepCu', 'daily']

vars_tower_state = ['theta','q','CO2']

if plot_diagnostics == True:
    for mode in modes: 
        
        print(mode)

        
        for var in variables.keys(): 
            
            print(var)
            
            #Define reference height based on variable type
            z_refs = (
                [50, 298] if var in vars_tower_state else
                [None]
            )
            
            #Go over each reference height
            for z_ref in z_refs: 
                
                print(f"z_ref: {z_ref} m")
                
                #OBS
                if mode != 'daily':
                    if var in vars_tower_state and var != 'CO2':
                        idx = testbed_functions.find_closest(all_heights, z_ref)
                        y_OBS, y_OBS_std = locals()[f'{mode}_tower_mean'][var][idx], locals()[f'{mode}_tower_std'][var][idx]

                    elif var == 'CO2':
                        z_str = f"{heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]}m"
                        y_OBS, y_OBS_std = locals()[f'{mode}_ghg_mean']['CO2'][z_str], locals()[f'{mode}_ghg_std']['CO2'][z_str]

                    elif var == 'h':
                        y_OBS, y_OBS_std = locals()[f'abl_{mode}_mean'], locals()[f'abl_{mode}_std']
                        if mode == 'SCu':
                            y_OBS_radiosonde = np.nanmean(SCu_values_h, axis=0)
                        elif mode == 'DeepCu':
                            y_OBS_radiosonde = np.nanmean(DeepCu_values_h, axis=0)

                #Load daily observations
                elif mode == 'daily':
                    if var in vars_tower_state and var != 'CO2':
                        y_OBS = daily_tower[var][testbed_functions.find_closest(all_heights, z_ref)]

                    elif var == 'CO2':
                        z_str = f"{heights_ghg[testbed_functions.find_closest(np.array(heights_ghg), z_ref)]}m"
                        y_OBS = daily_ghg['CO2'][z_str]

                    elif var == 'h':
                        y_OBS = abl_daily


                #DALES
                if mode == 'SCu': 
                    if var in vars_tower_state:
                        idx = testbed_functions.find_closest(z_DALES, z_ref)
                        y_DALES = DALES_profiles[var][:, idx]
                        y_DALES_no_clouds = DALES_profiles_no_clouds[var][:, idx]
                        
                    elif var == 'h':
                        y_DALES = DALES_tmser['abl_1D']
                        y_DALES_no_clouds = DALES_tmser_no_clouds['abl_1D']
                
                #CLASS
                if mode == 'SCu':
                    if var == 'theta':
                        y_CLASS = SCu_run.out.theta
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.theta
                    elif var == 'q':
                        y_CLASS = SCu_run.out.q*1000
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.q*1000
                    elif var == 'CO2':
                        y_CLASS = SCu_run.out.CO2
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.CO2
                    elif var == 'h':
                        y_CLASS = SCu_run.out.h
                        y_CLASS_no_clouds = SCu_run_no_clouds.out.h

                
                #-----------------------------------------------------------------------
                
                #Calculate DXR
                
                if mode!= 'daily':
                    
                    #OBS
                    DXR_OBS = testbed_functions.calc_amplitude(y_OBS)
                    
                    #DALES
                    if mode == 'SCu':
                        DXR_DALES = testbed_functions.calc_amplitude(y_DALES)
                        DXR_DALES_no_clouds = testbed_functions.calc_amplitude(y_DALES_no_clouds)
                    
                    #CLASS
                    if mode == 'SCu':
                        DXR_CLASS = testbed_functions.calc_amplitude(y_CLASS)
                        DXR_CLASS_no_clouds = testbed_functions.calc_amplitude(y_CLASS_no_clouds)
                        
                    #IFS        
                    
                    #Details
                    label = variables[var]['label']
                    step = variables[var]['step']
                    varname = variables[var]['varname']
                    datatype = variables[var]['datatype']
                    multiplier = variables[var]['multiplier']

                    IFS_data_y = []
                    x_IFS = testbed_functions.time_to_hour_fraction(pd.date_range("06:00", "18:00", freq="60min").time)
                    for key in resolutions.keys():
                            
                    
                        res = resolutions[key][0]
                        IFS_colour = resolutions[key][1]

                        if mode !='daily':
                                    
                            if datatype == 'sfc':
                                ds_base = xr.open_dataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_sfc_subset_km_Amazon_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude'])
                                
                                ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                
                                if mode == 'SCu':
                                    sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                                else:
                                    sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                                
                                sub_mean = sub.groupby('time.hour').mean(dim='time')
                                sub_std = sub.groupby('time.hour').std(dim='time')
                                
                                #Rename
                                sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                                sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                                
                                #ds
                                ds = xr.merge([sub_mean, sub_std])
                                ds = ds[sorted(ds.data_vars)]
                                
                                #Selection of hours
                                ds =  ds.sel(hour=slice(6, 18))
                            
                            else:
                                ds_base = xr.open_mfdataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_ml_subset_z_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_ref)

                                ds_base['time'] = pd.to_datetime(ds_base['time']) + timedelta(hours=-4)
                                
                                if mode == 'SCu':
                                    sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(SCu_list)) 
                                else:
                                    sub = ds_base.sel(time=ds_base['time'].dt.strftime('%Y-%m-%d').isin(DeepCu_list))
                                
                                sub_mean = sub.groupby('time.hour').mean(dim='time')
                                sub_std = sub.groupby('time.hour').std(dim='time')
                                
                                #Rename
                                sub_mean = sub_mean.rename({var: f"{var}_mean" for var in sub_mean.data_vars})
                                sub_std = sub_std.rename({var: f"{var}_std" for var in sub_std.data_vars})
                                
                                #ds
                                ds = xr.merge([sub_mean, sub_std])
                                ds = ds[sorted(ds.data_vars)]
                                
                                #Selection of hours
                                ds =  ds.sel(hour=slice(6, 18))
                            
                            y_IFS = ds[f'{varname}_mean'].values*multiplier
                            y_IFS_std = ds[f'{varname}_std'].values
                            
                            IFS_data_y.append(y_IFS)
                        
                    DXR_IFS = []
                    for i in range(len(IFS_data_y)):
                        DXR_IFS.append(testbed_functions.calc_amplitude(IFS_data_y[i]))
                        
                elif mode == 'daily':
                    
                    
                    label = variables[var]['label']
                    step = variables[var]['step']
                    varname = variables[var]['varname']
                    datatype = variables[var]['datatype']
                    multiplier = variables[var]['multiplier']
                    
                    daily_freqs = {
                            **{v: '30min' for v in vars_tower_state if v != 'CO2'},
                            'CO2': '1min',
                            'h': '5min'
                        }

                    freq = daily_freqs.get(var)
                    if freq:
                        x_OBS = testbed_functions.daily_range(date_range, freq, end="18:00")
                        if var == 'h':
                            x_OBS = x_OBS[~x_OBS.strftime('%H.%M').isin(['06.00'])]

                    df = pd.DataFrame({'time': x_OBS,
                                    'OBS': y_OBS})
                    
                    #Split date and time
                    df['date'] = df['time'].dt.date
                    df['time'] = df['time'].dt.time
                    
                    df['date'] = pd.to_datetime(df['date'])
                                
                    day_DXR = []
                    for date in date_range: 
                        
                        date = date.strftime('%Y-%m-%d')
                    
                        sel = df.loc[df['date'] == date]
                        
                        DXR = testbed_functions.calc_amplitude(sel['OBS'].values)
                        day_DXR.append(DXR)
                    
                    if mode == 'daily' and var == 'CO2' and z_ref == 298:
                        day_DXR = day_DXR[1:]  # Remove the first day as this is a NAN
                    
                    DXR_OBS_mean = np.nanmean(day_DXR)
                    DXR_OBS_std = np.nanstd(day_DXR)
                    DXR_OBS = day_DXR
                                    
                    #IFS
                    DXR_IFS_mean = []
                    DXR_IFS_std = []
                    DXR_IFS_day = []
                    
                    for res in resolutions.keys():
                        
                    
                        res = resolutions[key][0]
                        IFS_colour = resolutions[key][1]

                        if datatype == 'sfc':
                            ds = xr.open_mfdataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_sfc_subset_km_Amazon_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude'])
                        else:
                            ds = xr.open_mfdataset(f'{rootdir_IFS}/{res}/{datatype.upper()}/Processed/{res}_ml_subset_z_2022*.nc', engine='h5netcdf').mean(dim=['latitude','longitude']).sel(z=z_ref)

                        ds['time'] = pd.to_datetime(ds['time'].values) - pd.Timedelta(hours=4)  # Adjust for UTC-4

                        ds = ds.sel(time=slice('2022-08-09 06:00:00', '2022-08-18 18:00:00'))
                                
                        x_IFS = ds.time.values
                        y_IFS = ds[varname].values

                        indices = [list(x_IFS).index(item) for item in list(x_IFS) if item in list(x_OBS)]

                        y_IFS = y_IFS[indices]*multiplier
                        x_IFS = x_IFS[indices]

                        #Add a nan value each day at 19:00 LT 
                        for i in range(len(x_IFS)):
                            
                            datetimer = pd.to_datetime(x_IFS[i])
                            
                            if datetimer.hour == 18:
                                y_IFS = np.insert(y_IFS, i+1, np.nan)
                                x_IFS = np.insert(x_IFS, i+1, datetime(datetimer.year, datetimer.month, datetimer.day, 19, 0, 0))

                        #Dataframe
                        df = pd.DataFrame({'dates': x_IFS,
                                            'IFS': y_IFS})

                        #split date and time
                        df['time'] = pd.to_datetime(df['dates']).dt.time
                        df['date'] = pd.to_datetime(df['dates']).dt.date

                        df['date'] = pd.to_datetime(df['date'])

                        day_DXR = []
                        for date in date_range:
                            
                            date = date.strftime('%Y-%m-%d')
                            
                            sel = df.loc[df['date'] == date]
                            
                            DXR = testbed_functions.calc_amplitude(sel['IFS'].values)
                            day_DXR.append(DXR)

                        DXR_IFS_mean.append(np.nanmean(day_DXR))
                        DXR_IFS_std.append(np.nanstd(day_DXR))
                        DXR_IFS_day.append(day_DXR)

                        ds.close()
                                        
                #-------------------------------------------------- V I S U A L S E  -------------------------------------------------

                #Define a colour for the plot
                colour = 'tab:red' if mode == 'SCu' else 'tab:blue'

                
                #Load figure
                fig, ax = plt.subplots(figsize=(8.27, 11.69/2),dpi=300)
        
                #Visualise DXR
                if mode == 'SCu':
                    ax.bar(np.arange(0,8,1),[DXR_OBS,DXR_CLASS,DXR_CLASS_no_clouds,DXR_DALES,DXR_DALES_no_clouds,DXR_IFS[0],DXR_IFS[1],DXR_IFS[2]], color = colour)
                    ax.set_xticks(np.arange(0,8,1),['OBS','CLASS','CLASS NO CLOUDS','DALES', 'DALES NO CLOUDS', 'IFS ia61 (28km)', 'IFS ia62 (9km)', 'IFS ia63 (4.4km)'])
                    ax.axhline(DXR_OBS, color='k', linestyle='--', linewidth=1.5, label='DXR OBS')       
                    
                elif mode == 'DeepCu':
                    ax.bar(np.arange(0,4,1),[DXR_OBS,DXR_IFS[0],DXR_IFS[1],DXR_IFS[2]], color = colour)
                    ax.set_xticks(np.arange(0,4,1),['OBS','IFS ia61 (28km)', 'IFS ia62 (9km)', 'IFS ia63 (4.4km)'])
                    ax.axhline(DXR_OBS, color='k', linestyle='--', linewidth=1.5, label='DXR OBS')
                
                elif mode == 'daily':
                    ax.boxplot([DXR_OBS,DXR_IFS_day[0],DXR_IFS_day[1],DXR_IFS_day[2]])
                    ax.set_xticks(np.arange(1,4+1,1),['OBS','IFS ia61 (28km)', 'IFS ia62 (9km)', 'IFS ia63 (4.4km)'])

                #Detailing
                ax.set_ylabel(f'DXR | {label}')
                
                ymin, ymax = ax.get_ylim()
                ymin = step * np.round(ymin / step)
                ymax = step * np.round(ymax / step)
                ymax = ymax + step
                ymin = 0

                
                ax.set_yticks(np.arange(np.floor(ymin / step) * step, np.ceil(ymax / step) * step + step, step))

                ax.set_ylim(ymin,ymax)    


                ax.tick_params(axis='x')
                ax.tick_params(axis='y')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                ax.grid(linestyle=':',color='tab:gray',alpha=0.3,linewidth=axiswidth*2)
                
                ax.tick_params(which='major')
                ax.tick_params(which='minor')
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                text_ref = 'Shallow Convective Regime (ShCu)' if mode == 'SCu' else 'Shallow-to-Deep Convective Regime (ShDeep)' if mode == 'DeepCu' else 'Daily'

                if var!= 'cloud cover' and var != 'h' and var != 'mass flux':
                    fig.suptitle(f'{text_ref} | z = {z_ref} m',fontsize=plt.rcParams['font.size']*0.9, y=1)
                else:
                    fig.suptitle(f'{text_ref}',fontsize=plt.rcParams['font.size']*0.9, y=1)

                plt.tight_layout()

                #Export
                if not os.path.exists(f"{outdir}/{var}/"):
                    os.makedirs(f"{outdir}/{var}/")
                
                plt.savefig(f"{outdir}/{var}/{var}_{mode}_DXR_{z_ref}m.pdf",bbox_inches='tight')
                plt.savefig(f"{outdir}/{var}/{var}_{mode}_DXR_{z_ref}m.jpg",bbox_inches='tight',dpi=300)
                #plt.savefig(f"{outdir}/{var}/{var}_{mode}_DXR_{z_ref}m.svg",bbox_inches='tight')
                
                #plt.show()
                plt.close()