import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement


import pandas as pd
import matplotlib
import statsmodels
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
import scipy
from scipy import stats
#import hydroeval as he
#import pyswarms as ps
#from pyswarm import pso
import random
from ordered_set import OrderedSet
import warnings
warnings.filterwarnings('ignore')

# Change working directory
os.chdir('/Users/michellenguyen/Downloads/calibration_example 2') # change working directory

wd=getcwd()

from src.soils import *
from src.calibration_old import *

with open(wd + '/data/input_dict.pickle', 'rb') as input_data: 
    input_dict = pickle.load(input_data)

# Load the necessary CSV files and pickled data
df_aqua_units = pd.read_csv('/Users/michellenguyen/Downloads/PyCHAMP/examples/Aquacrop/Outputs/df_aqua_units_20240818_222323.csv')
planting_date = pd.read_csv(wd + '/data/CropPlantingDate_GMD4_WNdlovu_072423.csv')
bias_correction = pd.read_csv(wd + '/data/CropBiasCorrectionParams_GMD4_Wndlovu_062424.csv')
defaults = pd.read_csv(wd + '/data/CropDefaultParams_GMD4_Wndlovu_062424.csv')
defaults = defaults[defaults["Crop"] == 'Maize']

# Filter out rows where crop is "others" or irrig_method is 0
df_aqua_units = df_aqua_units[~((df_aqua_units['crop'] == 'others') | (df_aqua_units['irrig_method'] == 0))]

irrig_crop = '1' # code for irrigated corn
irrig_crop_dict = {k:v for (k,v) in input_dict.items() if irrig_crop in k}

calibration = list(irrig_crop_dict.items())
#

# getting the different datasets
gridmet = pd.concat([sublist[1][0] for sublist in calibration]) # data stored in nested list with 0 having the 
et = pd.concat([sublist[1][1] for sublist in calibration])
soil_irrig = pd.concat(SoilCompart([sublist[1][2] for sublist in calibration]))



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

# Initialize an empty DataFrame to store results
pyaqua_df = pd.DataFrame()

# Loop through each bid
for index, row in df_aqua_units.iterrows():
    bid = row['bid']
    maxirr_season = row['maxirr_season']
    irrig_method = row['irrig_method']
    
    # Update the defaults DataFrame with values from df_aqua_units
    defaults.loc[:, 'maxirr_season'] = maxirr_season
    defaults.loc[:, 'irrig_method'] = irrig_method
    
    # Prepare the gridmet data for the specific year and bid
    gridmet_data = gridmet[(gridmet['Year'] == row['year']) & (gridmet['crop_mn_codeyear'] == '1_Cheyenne')]
    
    # Run the AquaCrop model with updated parameters
    result_df = RunModelBiasCorrectedPyCHAMP(defaults, planting_date, gridmet_data, soil_irrig, bias_correction, 4, 11)
    
    # Add additional columns like bid and year to the result_df
    result_df['bid'] = bid
    result_df['year'] = row['year']
    
    # Append the result to the example_df
    pyaqua_df = pd.concat([pyaqua_df, result_df], ignore_index=True)
    

# Save the results to a CSV file
absolute_path = '/Users/michellenguyen/Downloads/pyaqua_df_full.csv'
pyaqua_df.to_csv(absolute_path, index=False)

print("done!")