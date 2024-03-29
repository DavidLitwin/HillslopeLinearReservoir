

import numpy as np
from landlab import RasterModelGrid
from landlab.grid.mappers import map_max_of_node_links_to_node
from landlab.utils import return_array_at_node

class SimpleHillslopeReservoirModel:
    """docstring for SimpleHillslopeReservoirModel."""

    def __init__(self, grid, hydraulic_conductivity=0.005):

        self._grid = grid

        # Create fields:

        if "topographic__elevation" in self._grid.at_node:
            self.elev = self._grid.at_node["topographic__elevation"]
        else:
            self.elev = self._grid.add_ones("node", "topographic__elevation")

        if "aquifer_base__elevation" in self._grid.at_node:
            self.base = self._grid.at_node["aquifer_base__elevation"]
        else:
            self.base = self._grid.add_zeros("node", "aquifer_base__elevation")

        if "effective_saturated__thickness" in self._grid.at_node:
            self.Hs = self._grid.at_node["effective_saturated__thickness"]
        else:
            self.Hs = self._grid.add_zeros("node", "effective_saturated__thickness")

        if "linear_reservoir_constant" in self._grid.at_node:
            self.K = self._grid.at_node["linear_reservoir_constant"]
        else:
            self.K = self._grid.add_zeros("node", "linear_reservoir_constant")

        if "groundwater__discharge" in self._grid.at_node:
            self.Qgw = self._grid.at_node["groundwater__discharge"]
        else:
            self.Qgw = self._grid.add_zeros("node", "groundwater__discharge")

        if "surface_water__discharge" in self._grid.at_node:
            self.Qsw = self._grid.at_node["surface_water__discharge"]
        else:
            self.Qsw = self._grid.add_zeros("node", "surface_water__discharge")

        self.Ksat = return_array_at_node(grid, hydraulic_conductivity)
        self.H = self.elev - self.base
        self.S = grid.calc_grad_at_link(self.elev)
        self.S_node = map_max_of_node_links_to_node(grid,self.S)
        # subsurface transport capacity
        self.Qgw_total_capacity = self.Ksat*self.H*self.S_node*grid.dx

    def run_one_step(self,intensity,duration):

        if intensity > 0.0:
            self.Qgw_capacity = self.Ksat*self.Hs*self.S_node*self._grid.dx
            self.Q = self._grid.at_node['drainage_area']*intensity
            self.Qsw = np.maximum(self.Q-self.Qgw_capacity,0)
            self.Qgw = np.minimum(self.Qgw_total_capacity,
                                            self.Qgw_capacity+self.Q)

        else:
            self.Qgw = self.Qgw*np.exp(-duration*self.K)
            self.Qsw[:] = 0
            self.Hs = np.minimum(self.H,self.Qgw/(self.Ksat*self.S_node*self._grid.dx))

    def calc_shear_stress(self,n_manning):
        rho = 1000
        g = 9.81

        return rho*g*self.S_node *( (n_manning*self.Qsw/3600)/(self._grid.dx*np.sqrt(self.S_node)) )**(3/5)
