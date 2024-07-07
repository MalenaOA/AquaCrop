# terminal:
# tar -xvzf C:\Users\m154o020\CHAMP\PyCHAMP\Summer2024\code_20240705\AquaCrop
# \cali_ex2.tar.gz -C C:\Users\m154o020\CHAMP\PyCHAMP\Summer2024\code_20240705\AquaCrop
# python.exe -m pip install --upgrade pip
# pip install git+https://github.com/MalenaOA/AquaCrop.git

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
#from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
import math
import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn.metrics as metrics
#from pyswarm import pso
import os
from os import chdir, getcwd
import statistics as stats
from scipy import stats
import hydroeval as he
#import pyswarms as ps
#from pyswarm import pso
import random
from ordered_set import OrderedSet
import warnings
warnings.filterwarnings('ignore')


os.chdir('/Users/michellenguyen/Downloads/calibration_example/') # change working directory

wd=getcwd()

from src.soils import *
from src.calibration_old import *

with open(wd + '/data/input_dict.pickle', 'rb') as input_data: 
    input_dict = pickle.load(input_data) 


#irrig_crop_calib = np.array(pd.read_csv(wd + '/eggs/data/sa_params/sa_results/gmd4_corn/gmd4_irrigated_corn_calibration_params_022324.txt',
                                       #sep=' ', header = None)) # full

planting_date = pd.read_csv(wd + '/data/CropPlantingDate_GMD4_WNdlovu_072423.csv')

yield_full = pd.read_csv(wd + '/data/Yield_GMD4_WNdlovu_v1_20230811.csv') 

irrig_depth = pd.read_csv(wd + '/data/IrrigationDepth_Updated_GMD4_WNdlovu_v2_20230123.csv') 

gdd = pd.read_csv(wd + '/data/CornGDD_GMD4_WNdlovu_20230827.csv') # growing degree days

# calibration_results
pso_fc = pd.read_csv(wd + '/data/calibration_results/gmd4_corn/gmd4_irrig_corn_pso_fc.csv')


defaults = pd.read_csv(wd + '/data/CropDefaultParams_GMD4_Wndlovu_022224.csv') # default model params
defaults = defaults[defaults["Crop"] == 'Maize']

print(defaults)

irrig_depth.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
irrig_depth.dropna(inplace=True)


irrig_crop = '1' # code for irrigated corn
irrig_crop_dict = {k:v for (k,v) in input_dict.items() if irrig_crop in k}

calibration = list(irrig_crop_dict.items())
#

# getting the different datasets
gridmet = pd.concat([sublist[1][0] for sublist in calibration]) # data stored in nested list with 0 having the 
et = pd.concat([sublist[1][1] for sublist in calibration])
soil_irrig = pd.concat(SoilCompart([sublist[1][2] for sublist in calibration]))


irrig_yield = yield_full[yield_full['Irrig_status'] == 'Irrigated']

#update 2022
planting_date = planting_date[planting_date["Crop"] == 'Maize']
planting_date['pdate'] = pd.to_datetime(planting_date['pdate'], format='%Y-%m-%d')
planting_date['har'] = pd.to_datetime(planting_date['har'], format='%Y-%m-%d')
planting_date['late_har'] = pd.to_datetime(planting_date['late_har'], format='%Y-%m-%d')

# Convert the 'date' column back to a new column in YMD format
planting_date['pdate'] = planting_date['pdate'].dt.strftime('%Y/%m/%d')
planting_date['har'] = planting_date['har'].dt.strftime('%Y/%m/%d')
planting_date['late_har'] = planting_date['late_har'].dt.strftime('%Y/%m/%d')

# add the ccx to cgc ratio for corn
planting_date['canopy'] = 0.96/0.012494 # change for different crops

empty_df = pd.DataFrame(data = {'variable': [],
                                'value': [],
                                'calib_val': []})

# order influential variables to match the order of the values
influ_var = np.array(list(OrderedSet(pso_fc['variable']))).reshape(-1, 1)
#influ_var


pso_fc_list = [pso_fc[pso_fc['iteration'] == i]['value'].tolist() for i in pso_fc['iteration'].unique()]
#[pso_fc_list[3]]


# full model
unlimited_model = RunAllCounties([pso_fc_list[3]], # make nested list
                      defaults,
                      gdd,
                      planting_date,
                      gridmet, #pychamp and needs unique identifiers 
                      soil_irrig,
                      'FC',
                      influ_var,
                      irrig_yield,
                      irrig_depth,
                      et,
                      1,  # irrigation method (0 = rainfed, 1 = irrigated)
                      4, # beginning of growth period (ET filter)
                      11, # end of growth period (ET filter)
                      'train')

g = sns.FacetGrid(unlimited_model[0][0], col='County', margin_titles=True, col_wrap=3, height=4, aspect=1.5)
sns.set(style="white", font_scale=3)

# Line plots
g.map_dataframe(sns.lineplot, x="Year", y="Calib Yield (t/ha)", label='Sim', color='grey', linewidth=3.5)
g.map_dataframe(sns.lineplot, x="Year", y="YieldUSDA", label='Obs (RMA/USDA-NASS)', color='#2e8b57', linestyle='dashed', linewidth=5)




g.set_titles(col_template="{col_name}", size=70)
g.set_axis_labels("", "", size=23)
g.set_titles("{col_name}", fontsize=72)
g.tight_layout()

# Set x-axis ticks
for ax in g.axes.flat:
    ax.tick_params(axis="x", size=25, direction='out')
    ax.set_xticks([2010, 2015, 2020])
    ax.tick_params(axis='both', which='major', labelsize=25)

# Adjust space
#g.fig.subplots_adjust(top=0.5)

# Add legend and y-axis label
g.add_legend(fontsize=32)

plt.subplots_adjust(bottom = .15, left=0.065)
g.fig.text(0.015, 0.5, 'Yield (t/ha)', va='center', rotation='vertical', fontsize=46, weight='bold')

plt.show()