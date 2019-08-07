import numpy as np
from landlab import RasterModelGrid

def calc_width_exponent(grid):
    w_bar = grid.at_node['average_width']
    w_bar_below = w_bar[grid.at_node['flow__receiver_node']]

    w_bar[np.isnan(w_bar)] = grid.dx
    w_bar_below[np.isnan(w_bar_below)] = grid.dx

    w_bar[w_bar<=0.0] = grid.dx
    w_bar_below[w_bar_below <= 0.0] = grid.dx

    #really aught to calculate the length of the link, and divide by that, not just dx.
    #lengths = grid.length_of_link[grid.at_node['flow__link_to_receiver_node']]

    return (np.log(w_bar_below)-np.log(w_bar))/grid.dx


def calc_z_eff(grid,p,Tr,Td,sf,sw,ETmax):
    #mean rainfall rate, when rain occurs
    p_bar = p/Tr

    #truncated distribution for f, because of hortonian overland flow
    f_mean = p_bar*(1-np.exp(-grid.at_node['hydraulic_conductivity'][grid.core_nodes]/p_bar))

    #long time mean infiltration rate.
    f_mean_uniform = f_mean*Tr/(Tr+Td)

    #mean et flux, assuming mean soil moisture is sf (seems to be a safe bet with current model setup)
    qet_mean = (sf-sw)/(1-sw)*ETmax

    #mean qs+qr
    q_mean = (f_mean_uniform - qet_mean)*grid.at_node['drainage_area'][grid.core_nodes]

    return q_mean/(grid.at_node['hydraulic_conductivity'][grid.core_nodes]*grid.at_node['hydraulic_conductivity_f'][grid.core_nodes]*grid.at_node['average_width'][grid.core_nodes])

def calc_res_constant(grid):
    dzdx_avg = grid.at_node['catchment_average__slope'][grid.core_nodes]
    alpha = np.arctan(dzdx_avg)
    Ksat = grid.at_node['hydraulic_conductivity'][grid.core_nodes]
    K_f = grid.at_node['hydraulic_conductivity_f'][grid.core_nodes]
    a = grid.at_node['width_exponent'][grid.core_nodes]
    zeff = grid.at_node['effective__depth'][grid.core_nodes]
    L = grid.at_node['catchment_average_flow__length'][grid.core_nodes]
    n = grid.at_node['porosity'][grid.core_nodes]

    #note that the Berne paper uses L/2, but here we actually have the average
    #length L, so 2 is left out of the numerator.
    return (Ksat*K_f*(np.sin(alpha)- a*zeff*np.cos(alpha)))/(L*n)
