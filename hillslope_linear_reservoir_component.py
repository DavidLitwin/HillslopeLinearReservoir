# -*- coding: utf-8 -*-
"""
HillslopeLinearReservoir Component

@author: D Litwin
"""


import numpy as np
from landlab import Component, RasterModelGrid
from landlab.utils.flow__distance import calculate_flow__distance
from landlab.components.flow_accum import flow_accumulation

def _LeakageLoss(S,f,Qet,Qh,Qs,Qrc,dt,Sf,St,Ksat,A,vo,Zo,w):

    if S + f*A*dt - Ksat*A*dt - Qet >= Sf  and Qrc+Qh+Qs < vo*Zo*w: #if saturated above field capacity + additional volume added that timestep (incuding f=0), and there is no outlet export limit, then drain at infiltration rate
        Qr = Ksat*A

    elif S + f*A*dt - Qet*dt >= Sf  and S + f*A*dt - Qet*dt - Ksat*A*dt < Sf and Qrc+Qh+Qs < vo*Zo*w:
        Qr = (S + f*A*dt - Qet*dt - Sf)/dt

#        Qr = f*A #in this case, it only drains when there's infiltration...
#        Qr = (S-Sf)/(St-Sf)*Ksat #a simple linearized option, releases water too slowly?

    elif S + f*A*dt - Qet*dt - Qrc*dt >= Sf and Qrc+Qh+Qs >= vo*Zo*w: #if saturated above field capacity + additional volume added that timestep (incuding f=0), and there is outlet export limit, then drain at the lesser of the reservoir export rate and the rate required to return to field capacity
        Qr = Qrc

    elif S + f*A*dt - Qet*dt >= Sf  and S + f*A*dt - Qet*dt - Ksat*A*dt and Qrc+Qh+Qs >= vo*Zo*w:
        Qr = (S + f*A*dt - Qet*dt - Sf)/dt

    else: #if timestep starts below field capacity, assume no deep leakage that timestep
        Qr = 0

    return Qr

def _SaturationExcessLoss(S,Qr,f,St,Sf,A,dt):
    Qs = max(0,(S+f*A*dt-Qr*dt-St)/dt)

    #equivalently
#    if S + f*A*dt - Qr < St:
#        Qs = 0
#    elif S + f*A*dt - Qr > St:
#        Qs = (S+f*A*dt-Qr*dt-St)/dt

    return Qs

def _EvapotranspirationLoss(p,S,Sw,St,ETmax,A,dt):

    if p == 0.0 and S > Sw+(S-Sw)/(St-Sw)*ETmax*A*dt:
        Qet = (S-Sw)/(St-Sw)*ETmax*A
    elif p == 0.0 and (S-Sw)/(St-Sw)*ETmax*A > (S - Sw)/dt:
        Qet = (S - Sw)/dt
    else:
        Qet = 0

    return Qet

class HillslopeLinearReservoir(Component):
    """
    Simulate surface and subsurface flow and storage in independent reservoirs
    for catchments defined upslope of each node.

    The model uses a two-reservoir model, the first representing soil storage,
    while the second one is a linear groundwater reservoir.

    Parameters
    ----------
    grid: ModelGrid
            Landlab ModelGrid object
    hydraulic_conductivity: float, field name, or array of float
            saturated hydraulic conductivity, m/s
            Default = 0.01 m/s
    hydraulic_conductivity_f: float
            A scaling factor indicating how much more conductive the soil is
            in the lateral direction in compared to the vertical direction
            Default = 5.0
    field_capacity: float
            The relative soil moisture above which there is free drainage
            Default = 0.3
    wilting_point: float
            The relative soil mositure at the permanent wilting point
            Default = 0.18
    max_evapotranspiration: float
            The maximum rate of evapotranspiration, which occurs when relative
            soil moisture = 1.
            Default = 1e-3 m/hr
    regularization_f: float
            factor controlling the smoothness of the transition between
            surface and subsurface flow
    """

    _name = "HillslopeLinearReservoir"

    _input_var_names = set(("topographic__elevation",
                            "regolith_base__elevation"))

    _output_var_names = set(
        ("catchment_average_regolith__thickness","regolith__thickness", "topographic__gradient","catchment_average_topographic__gradient",
        "catchment_average__storage", "catchment_average__evapotranspiration"
         "groundwater__discharge","surface_water__discharge",
         "effective__depth","linear_reservoir__constant","")
    )

    _var_units = {
        "topographic__elevation": "m",
        "regolith_base__elevation": "m",
        "catchment_average_regolith__thickness": "m",
        "regolith__thickness": "m",
        "topographic__gradient":"m",
        "catchment_average_topographic__gradient": "m",
        "catchment_average__storage": "m3",
        "catchment_average__evapotranspiration": "m3/hr",
        "groundwater__discharge": "m3/hr",
        "surface_water__discharge": "m3/hr",
        "effective__depth": "m",
        "linear_reservoir__constant": "hr-1",

    }

    _var_mapping = {
        "topographic__elevation": "node",
        "regolith_base__elevation": "node",
        "catchment_average_regolith__thickness": "node",
        "regolith__thickness": "node",
        "topographic__gradient":"node",
        "catchment_average_topographic__gradient": "node",
        "catchment_average__storage": "node",
        "catchment_average__evapotranspiration": "node",
        "groundwater__discharge": "node",
        "surface_water__discharge": "node",
        "effective__depth": "node",
        "linear_reservoir__constant": "node",
    }

    _var_doc = {
        "topographic__elevation": "elevation of land surface",
        "regolith_base__elevation": "elevation of impervious layer",
        "catchment_average_regolith__thickness": "average thickness of permeable layer over catchment",
        "regolith__thickness": "thickness of permeable layer",
        "topographic__gradient":"topographic gradient",
        "catchment_average_topographic__gradient": "catchment average topographic gradient",
        "catchment_average__storage": "catchment average storage",
        "catchment_average__evapotranspiration": "catchment average et",
        "groundwater__discharge": "groundwater discharge at outlet node",
        "surface_water__discharge": "surface water discharge at outlet node",
        "effective__depth": "estimate of the effective depth for Boussinesq hillslope"
        "linear_reservoir__constant": "linear reservoir constant for catchment",
    }


    def __init__(self, grid, hydraulic_conductivity=0.5, hydraulic_conductivity_f=5,
    field_capacity=0.3, wilting_point=0.18, max_evapotranspiration=0.001, porosity=0.5,
    mean_storm_dt=1, mean_interstorm_dt=24, mean_storm_depth = 0.02 ):

    """Initialize the HillslopeLinearReservoir.

    Parameters
    ----------
    grid: ModelGrid
            Landlab ModelGrid object
    hydraulic_conductivity: float, field name, or array of float
            saturated hydraulic conductivity, m/s
            Default = 0.5 m/hr
    hydraulic_conductivity_f: float, or array of float
            A scaling factor indicating how much more conductive the soil is
            in the lateral direction in compared to the vertical direction
            Default = 5.0
    field_capacity: float
            The relative soil moisture above which there is free drainage
            Default = 0.3
    wilting_point: float
            The relative soil mositure at the permanent wilting point
            Default = 0.18
    max_evapotranspiration: float, or array of float
            The maximum rate of evapotranspiration, which occurs when relative
            soil moisture = 1.
            Default = 1e-3 m/hr
    porosity: float, or array of float
            The soil drainable porosity.
            Default = 0.5
    mean_storm_dt: float
            The mean duration of storm events, used to generate storms from an
            exponential distribution of durations
            Default = 1 hr
    mean_interstorm_dt: float
            The mean duration between storm events, used to generate interstorm
            periods from an exponential distribution of durations
            Default = 24 hr
    mean_storm_depth: float
            The mean depth of rainfall in a storm event.
            Default = 0.02 m


    """

        # Store grid
        self._grid = grid

        # Shorthand
        self.cores = grid.core_nodes
        self.inactive_links = np.where(grid.status_at_link != ACTIVE_LINK)[0]

        # Convert parameters to fields if needed, and store a reference
        self.Ksat = return_array_at_link(grid, hydraulic_conductivity)
        self.K_f = return_array_at_link(grid, hydraulic_conductivity_f)
        self.n = return_array_at_node(grid,porosity)
        self.sf = field_capacity
        self.sw = wilting_point
        self.ETmax = max_evapotranspiration

        self.Tr = mean_storm_dt
        self.Td = mean_interstorm_dt
        self.p = mean_storm_depth

        #create fields
        if "topographic__elevation" in self.grid.at_node:
            self.elev = self.grid.at_node["topographic__elevation"]
        else:
            self.elev = self.grid.add_ones("node", "topographic__elevation")

        if "regolith_base__elevation" in self.grid.at_node:
            self.base = self.grid.at_node["aquifer_base__elevation"]
        else:
            self.base = self.grid.add_zeros("node", "aquifer_base__elevation")

        if "effective__depth" in self.grid.at_node:
            self.zeff = self.grid.at_node["effective__depth"]
        else:
            self.zeff = self.grid.add_zeros("node", "effective__depth")
            self.zeff = self.calc_z_eff()

        if "linear_reservoir__constant" in self.grid.at_node:
            self.k = self.grid.at_node["linear_reservoir__constant"]
        else:
            self.k = self.grid.add_zeros("node", "linear_reservoir__constant")
            self.k = self.calc_res_constant()

        if "regolith__thickness" in self.grid.at_node:
            self.zr = self.grid.at_node["regolith__thickness"]
        else:
            self.zr = self.grid.add_zeros("node", "regolith__thickness")
            self.zr = self.elev-self.base

        if "topographic__gradient" in self.grid.at_node:
            self.dzdx = self.grid.at_node["topographic__gradient"]
        else:
            self.dzdx = self.grid.add_zeros("node", "topographic__gradient")
            self.dzdx = self.grid.calc_slope_at_node()

        if "flow__distance" in self.grid.at_node:
            self.L = self.grid.at_node["flow__distance"]
        else:
            self.L = calculate_flow__distance(self.grid, add_to_grid=True, noclobber=False)

        if "width__exponent" in self.grid.at_node:
            self.a = self.grid.at_node["width__exponent"]
        else:
            self.a = self.grid.add_zeros("node", "width__exponent")

        if "accumulated__area" in self.grid.at_node:
            self.area = self.grid.at_node["accumulated__area"]
        else:
            self.area = self.grid.add_zeros("node", "accumulated__exponent")

        if "catchment_average__storage" in self.grid.at_node:
            self.Savg = self.grid.at_node["catchment_average__storage"]
        else:
            self.Savg = self.grid.add_zeros("node", "catchment_average__storage")

        if "catchment_average_regolith__thickness" in self.grid.at_node:
            self.zr_avg = self.grid.at_node["catchment_average_regolith__thickness"]
        else:
            self.zr_avg = self.grid.add_zeros("node", "catchment_average_regolith__thickness")

        if "catchment_average_topographic__gradient" in self.grid.at_node:
            self.dzdx_avg = self.grid.at_node["catchment_average_topographic__gradient"]
        else:
            self.dzdx_avg = self.grid.add_zeros("node", "catchment_average_topographic__gradient")

        _area,_dzdx,_ = flow_accumulation(self.grid.at_node['flow__receiver_node'],
                                        node_cell_area=1.0, runoff_rate=self.dzdx)
        _,_zr,_ = flow_accumulation(self.grid.at_node['flow__receiver_node'],
                                        node_cell_area=1.0, runoff_rate=self.zr)

        self.zr_avg = _zr/_area
        self.Savg = self.area*self.zr_avg
        self.dzdx_avg = _dzdx/_area

        self.ET_avg = self.grid.add_zeros("node", "catchment_average__evapotranspiration")
        self.Qbc = self.grid.add_zeros("node", "groundwater__discharge")
        self.Qsc = self.grid.add_zeros("node", "surface_water__discharge")


    def calc_z_eff(self):
        w_bar = self._grid.at_node['Area']/self._grid.at_node['Length']

        #mean rainfall rate, when rain occurs
        p_bar = self.p/self.Tr

        #truncated distribution for f, because of hortonian overland flow
        f_mean = p_bar*(1-np.exp(-self.K/p_bar))

        #long time mean infiltration rate.
        f_mean_uniform = f_mean*self.Tr/(self.Tr+self.Td)

        #mean et flux, assuming mean soil moisture is sf (seems to be a safe bet with current model setup)
        qet_mean = (self.sf-self.sw)/(1-self.sw)*self.ETmax

        #mean qs+qr
        q_mean = (f_mean_uniform - qet_mean)*self._grid.at_node['Area']

        return q_mean/(self.K*self.K_f*w_bar)


    def calc_res_constant(self):
        alpha = np.arctan(self.dzdx_avg)
        return (2*self.Ksat*self.K_f*(np.sin(alpha)-
                        self.a*self.zeff*np.cos(alpha)))/(self.L*self.n)
