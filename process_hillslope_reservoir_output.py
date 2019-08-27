# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:42:15 2019

@author: dgbli
"""
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid
import pickle

#%% file paths for output files

base_path = "/Users/dgbli/Documents/Project_outputs/Landlab_runoff_LEM_output/"

files = glob.glob(base_path+'runoff_data*')

grid_path = "/Users/dgbli/Documents/Project_outputs/Landlab_runoff_LEM_output/model_grid.p"

#%% Grid data from run

grid_file = open(grid_path,'rb')
mg= pickle.load(grid_file)

#%% Calculate cumulate groundwater, surface water, and shear stress at each point

Qbc_cumulative = np.zeros(mg.number_of_nodes)
Qsc_cumulative = np.zeros(mg.number_of_nodes)
Taub_cumulative = np.zeros(mg.number_of_nodes)

start = time.process_time()
for file in files:
    
    runoff_data_file = open(file,'rb')
    node = int(''.join(filter(str.isdigit, file[-10:-2])))
    df = pickle.load(runoff_data_file)

    Qbc_cumulative[node] = np.sum(df['Qbc']*df['dt'])
    Qsc_cumulative[node] = np.sum(df['Qsc']*df['dt'])
    Taub_cumulative[node] = np.sum(df['Taub']*df['dt'])
    

run_duration = time.process_time()-start

#%% Add cumulative quantities to the grid

Qbc_cum = mg.add_zeros('node','cumulative__base_flow')
Qbc_cum[:] = Qbc_cumulative

Qsc_cum = mg.add_zeros('node','cumulative__surface_runoff')
Qsc_cum[:] = Qsc_cumulative

Taub_cum = mg.add_zeros('node','cumulative__shear_stress')
Taub_cum[:] = Taub_cumulative

#%% Plot cumulative quantities

plt.figure()
imshow_grid(mg,'cumulative__base_flow', colorbar_label='Cumulative baseflow $[m^3]$')

plt.figure()
imshow_grid(mg,'cumulative__surface_runoff', colorbar_label='Cumulative surface flow $[m^3]$')

plt.figure()
imshow_grid(mg,'cumulative__shear_stress', colorbar_label='Cumulative shear stress $[N/m^2 \, hr]$')

#%% Plots to look at timeseries at one node

#node = 

A = mg.at_node['drainage_area'][node]

#calculated cumulative fluxes
df['t'] = np.cumsum(df['dt'])
df['precip cumulative'] = np.cumsum(df['dt']*df['intensity'])
df['f cumulative'] =  np.cumsum(df['dt']*df['f'])
df['Qh cumulative'] =  np.cumsum(df['dt']*df['Qh'])
df['Qs cumulative'] =  np.cumsum(df['dt']*df['Qs'])
df['Qr cumulative'] =  np.cumsum(df['dt']*df['Qr'])
df['Qet cumulative'] =  np.cumsum(df['dt']*df['Qet'])
df['Qsc cumulative'] =  np.cumsum(df['dt']*df['Qsc'])
df['Qbc cumulative'] =  np.cumsum(df['dt']*df['Qbc'])


plt.figure(figsize=(8,6))
plt.plot(df['t']/24,df['Qet cumulative']+df['Qr cumulative']+df['Qs cumulative']+df['S']-df['S'].iloc[0],'b-',linewidth=3, alpha=0.5,label='Total Soil Fluxes +Storage')
plt.plot(df['t']/24,df['f cumulative']*A,'k:',label='infitration')
plt.plot(df['t']/24,df['Qs cumulative'],'b:',label='soil moisture excess')
plt.plot(df['t']/24,df['Qr cumulative'],'g:',label='deep infiltration')
plt.plot(df['t']/24,df['Qet cumulative'],'r:',label='evapotranspiration')
plt.ylabel('Cumulative Volume $[m^3]$')
plt.xlabel('Time [d]')
plt.legend(frameon=False)
#plt.savefig('Cumulative_soilmoisture_'+str(j)+'.png')


plt.figure(figsize=(8,6))
plt.plot(df['t']/24,df['S']-df['S'].iloc[0]+df['Qsc cumulative']+df['Qbc cumulative']+df['Qet cumulative'],'b-',linewidth=3, alpha=0.5,label='Total Catchment Fluxes + Storage')
plt.plot(df['t']/24,df['precip cumulative']*A,'k:',label='Precipitation')
plt.plot(df['t']/24,df['Qsc cumulative'],'b:',label='surface runoff')
plt.plot(df['t']/24,df['Qbc cumulative'],'g:',label='subsurface runoff')
plt.plot(df['t']/24,df['Qet cumulative'],'r:',label='evapotranspiration')
plt.ylabel('Cumulative Volume $[m^3]$')
plt.xlabel('Time [d]')
plt.legend(frameon=False)
#plt.savefig('Cumulative_outflow_'+str(j)+'.png')


#%% process new cumulative outputs

data_path = "/Users/dgbli/Documents/Project_outputs/Landlab_runoff_LEM_output/runoff_data_cumulative.p"
grid_path= "/Users/dgbli/Documents/Project_outputs/Landlab_runoff_LEM_output/model_grid.p"

grid_file = open(grid_path,'rb')
mg= pickle.load(grid_file)
grid_file.close()

data_file = open(data_path,'rb')
data = pickle.load(data_file)
data_file.close()

#%%
cores = mg.core_nodes
Qsc_cum = np.zeros(mg.number_of_nodes)
Qbc_cum = np.zeros(mg.number_of_nodes)
Taub_cum = np.zeros(mg.number_of_nodes)

for i in range(len(cores)):
    node = mg.core_nodes[i]
    Qbc_cum[node] = data[i][0]
    Qsc_cum[node] = data[i][1]
    Taub_cum[node] = data[i][2]
    
#%%
plt.figure()
imshow_grid(mg,np.log(Qbc_cum+1),colorbar_label='Log Cumulative groundwater discharge', cmap='Blues')

plt.figure()
imshow_grid(mg,np.log(Qsc_cum+1),colorbar_label='Log Cumulative surface water discharge',cmap='Oranges')

plt.figure()
imshow_grid(mg,np.log(Taub_cum+1),colorbar_label='Log Cumulative shear stress',cmap='viridis')






