# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:15:06 2019

@author: dgbli
"""
import time
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

from landlab import RasterModelGrid, FIXED_VALUE_BOUNDARY
from landlab.components import FlowAccumulator
from landlab.utils.flow__distance import calculate_flow__distance
from landlab.components.flow_accum.flow_accum_to_n import flow_accumulation_to_n
from landlab.components.uniform_precip import PrecipitationDistribution

from hillslope_reservoir_model import HillslopeReservoirModel
from hillslope_reservoir_grid_funcs import *

#%% Initialize grid and parameters

zr_uniform = 2 #m
n_uniform = 0.5 #[]
n_manning_uniform = 0.05 #[s/m^1/3]
Ksat_uniform = 0.5 #m/hr
K_f_uniform = 5 #[]
a_uniform = 0 #[]

p = 0.02 #m
Tr = 1 #hr
Td = 24 #hr
ETmax_uniform = 5E-4 #m/hr
sw_uniform = 0.18 #[]
sf_uniform = 0.3 #[]

boundaries = {'top': 'closed','bottom': 'closed','right':'closed','left':'closed'}
mg = RasterModelGrid((51, 51), spacing=10.0,bc=boundaries)
mg.status_at_node[1] = FIXED_VALUE_BOUNDARY

#%% Calculate grid attributes

z = mg.add_zeros('node', 'topographic__elevation')
z[:] = (0.001*mg.x_of_node**2 + 0.001*mg.y_of_node**2)+2

zr = mg.add_zeros('node','regolith__thickness')
zr[mg.core_nodes] = zr_uniform

base = mg.add_zeros('node','aquifer_base__elevation')
base[mg.core_nodes] = z[mg.core_nodes] - zr[mg.core_nodes]

n = mg.add_zeros('node','porosity')
n[mg.core_nodes] = n_uniform

n_manning = mg.add_zeros('node','manning_n')
n_manning[mg.core_nodes] = n_manning_uniform

Ksat = mg.add_zeros('node','hydraulic_conductivity')
Ksat[mg.core_nodes] = Ksat_uniform

K_f = mg.add_zeros('node','hydraulic_conductivity_f')
K_f[mg.core_nodes] = K_f_uniform

sw = mg.add_zeros('node','soil_wilting_point')
sw[mg.core_nodes] = sw_uniform

sf = mg.add_zeros('node','soil_field_capacity')
sf[mg.core_nodes] = sf_uniform

ETmax = mg.add_zeros('node','Maximum_ET_rate')
ETmax[mg.core_nodes] = ETmax_uniform

a = mg.add_zeros('node','width_exponent')


fa = FlowAccumulator(mg, 'topographic__elevation',flow_director='MFD')
fa.run_one_step()

dzdx = mg.calc_slope_at_node()
L = calculate_flow__distance(mg, add_to_grid=True, noclobber=False)

receiver = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']

_areas,_dzdx,node_order = flow_accumulation_to_n(receiver,proportions,node_cell_area=1.0, runoff_rate=dzdx)
_,_zr,_ = flow_accumulation_to_n(receiver,proportions,node_cell_area=1.0, runoff_rate=zr)
_,_L,_ = flow_accumulation_to_n(receiver,proportions,node_cell_area=1.0, runoff_rate=L)

# average slope is the accumulated slopes over the accumulated areas
dzdx_avg = mg.add_field('catchment_average__slope',_dzdx/_areas,at='node')

# similarly for average regolith thickness
zr_avg = mg.add_field('catchment_average_regolith__thickness',_zr/_areas,at='node')

# catchment total storage is that average thickness times the area
S_avg = mg.add_field('catchment_average__storage',zr_avg*mg.at_node['drainage_area'],at='node')

# the average flow length to the outlet is the accumulated flow lengths from each cell
# to the global outlet minus the distance from the local outlet to the global outlet
L_avg = mg.add_field('catchment_average_flow__length',_L/_areas-L+mg.dx,at='node')

zeff = mg.add_zeros('node','effective__depth')

# the reservoir constant is the advective timescale of a Boussinesq aquifer, from Berne et al 2003. Here width_exponent = 0.
kres = mg.add_zeros('node','linear_reservoir_constant')
kres[mg.core_nodes] = calc_res_constant(mg)


#%% Pickle grid for later use

base_path = "/Users/dgbli/Documents/Project_outputs/Landlab_runoff_LEM_output/"

file = open(base_path+'model_grid.p', 'wb')
pickle.dump(mg, file)
file.close()

#%% generate storm timeseries
T = 24*365
dt = np.min(np.floor(1/mg.at_node['linear_reservoir_constant'][mg.core_nodes]))

precip = PrecipitationDistribution(mean_storm_duration=Tr, mean_interstorm_duration=Td,
                                               mean_storm_depth=p, total_t=T, delta_t=dt)
durations = []
intensities = []
precip.seed_generator(seedval=1)
for (interval_duration, rainfall_rate_in_interval) in (
                precip.yield_storm_interstorm_duration_intensity(subdivide_interstorms=True)
):
   durations.append(interval_duration)
   intensities.append(rainfall_rate_in_interval)
N = len(durations)

#%% Non-parallel version for running all models at nodes


#for i in range(mg.number_of_core_nodes):
#
#    start = time.process_time()
#
#
#    node = mg.core_nodes[i]
#
#    df = pd.DataFrame(columns = ['dt', 'intensity', 'f', 'Qh', 'Qet', 'Qr', 'Qs', 'S', 'Qrc', 'Qbc', 'Qsc', 'Taub'])
#
#    hrm = HillslopeReservoirModel(mg,node,sw=sw,sf=sw,ETmax=ETmax)
#
#    for j in range(N):
#        hrm.run_one_step(durations[j],intensities[j])
#
#        df.loc[j] = hrm.yield_all_timestep_data()
#
#    file = open(base_path+'runoff_data_'+str(node)+'.p', 'wb')
#    pickle.dump(df, file)
#    file.close()
#
#    print(time.process_time()-start)
#
#    if i>10:
#        break

#%% Parallel version for running all models at nodes

def run_one_model(i):
    node = mg.core_nodes[i]

    Qbc_cum = 0
    Qsc_cum = 0
    Taub_cum = 0
    # df = pd.DataFrame(columns = ['dt', 'intensity', 'f', 'Qh', 'Qet', 'Qr', 'Qs', 'S', 'Qrc', 'Qbc', 'Qsc', 'Taub'])

    hrm = HillslopeReservoirModel(mg,node)

    for j in range(N):
        hrm.run_one_step(durations[j],intensities[j])
        Qbc_cum += hrm.Qbc*durations[j]
        Qsc_cum += hrm.Qsc*durations[j]
        Taub_cum += hrm.Taub*durations[j]


    return [Qbc_cum,Qsc_cum,Taub_cum]

    #     df.loc[j] = hrm.yield_all_timestep_data()
    #
    # file = open(base_path+'runoff_data_'+str(node)+'.p', 'wb')
    # pickle.dump(df, file)
    # file.close()


def run_one_model_precip(i):
    node = mg.core_nodes[i]

    #precip
    T = 24*365
    dt = min(np.floor(1/mg.at_node['linear_reservoir_constant'][node]),12)

    precip = PrecipitationDistribution(mean_storm_duration=Tr, mean_interstorm_duration=Td,
                                                   mean_storm_depth=p, total_t=T, delta_t=dt)
    durations = []
    intensities = []
    precip.seed_generator(seedval=1)
    for (interval_duration, rainfall_rate_in_interval) in (
                    precip.yield_storm_interstorm_duration_intensity(subdivide_interstorms=True)
    ):
       durations.append(interval_duration)
       intensities.append(rainfall_rate_in_interval)
    N = len(durations)


    df = pd.DataFrame(columns = ['dt', 'intensity', 'f', 'Qh', 'Qet', 'Qr', 'Qs', 'S', 'Qrc', 'Qbc', 'Qsc', 'Taub'])

    hrm = HillslopeReservoirModel(mg,node)

    for j in range(N):
        hrm.run_one_step(durations[j],intensities[j])

        df.loc[j] = hrm.yield_all_timestep_data()

    file = open(base_path+'runoff_data_'+str(node)+'.p', 'wb')
    pickle.dump(df, file)
    file.close()




if __name__ == '__main__':
    p = mp.Pool()
    output = p.map(run_one_model, range(mg.number_of_core_nodes))

    file = open(base_path+'runoff_data_cumulative.p', 'wb')
    pickle.dump(output, file)
    file.close()


# start = time.process_time()
# run_duration = time.process_time()-start
# file = open(base_path+'time.txt', 'w')
# file.write('duration (s)')
# file.write(str(run_duration))
# file.close()
