
import numpy as np
from landlab import RasterModelGrid
import multiprocessing as mp

def LeakageLoss(S,f,Qet,Qh,Qs,Qrc,dt,Sf,St,Ksat,A,vo,Zo,w):

    #if saturated above field capacity + additional volume added that timestep
    # (incuding f=0), and there is no outlet export limit, then drain at infiltration rate
    if S + f*A*dt - Ksat*A*dt - Qet >= Sf  and Qrc+Qh+Qs < vo*Zo*w:
        Qr = Ksat*A

    elif S + f*A*dt - Qet*dt >= Sf  and S + f*A*dt - Qet*dt - Ksat*A*dt < Sf and Qrc+Qh+Qs < vo*Zo*w:
        Qr = (S + f*A*dt - Qet*dt - Sf)/dt

#        Qr = f*A #in this case, it only drains when there's infiltration...
#        Qr = (S-Sf)/(St-Sf)*Ksat #a simple linearized option, releases water too slowly?

    #if saturated above field capacity + additional volume added that timestep
    # (incuding f=0), and there is outlet export limit, then drain at the lesser
    # of the reservoir export rate and the rate required to return to field capacity
    elif S + f*A*dt - Qet*dt - Qrc*dt >= Sf and Qrc+Qh+Qs >= vo*Zo*w:
        Qr = Qrc

    elif S + f*A*dt - Qet*dt >= Sf  and S + f*A*dt - Qet*dt - Ksat*A*dt and Qrc+Qh+Qs >= vo*Zo*w:
        Qr = (S + f*A*dt - Qet*dt - Sf)/dt

    #if timestep starts below field capacity, assume no deep leakage that timestep
    else:
        Qr = 0

    return Qr

def SaturationExcessLoss(S,Qr,f,St,Sf,A,dt):
    Qs = max(0,(S+f*A*dt-Qr*dt-St)/dt)

    #equivalently
#    if S + f*A*dt - Qr < St:
#        Qs = 0
#    elif S + f*A*dt - Qr > St:
#        Qs = (S+f*A*dt-Qr*dt-St)/dt

    return Qs

def EvapotranspirationLoss(p,S,Sw,St,ETmax,A,dt):

    if p == 0.0 and S > Sw+(S-Sw)/(St-Sw)*ETmax*A*dt:
        Qet = (S-Sw)/(St-Sw)*ETmax*A
    elif p == 0.0 and (S-Sw)/(St-Sw)*ETmax*A > (S - Sw)/dt:
        Qet = (S - Sw)/dt
    else:
        Qet = 0

    return Qet


class HillslopeReservoirModel:

    def __init__(self,grid,node):

        self.rho = 1000
        self.g = 9.81

        self.A = grid.at_node['drainage_area'][node]
        self.w = grid.dx
        self.n = grid.at_node['porosity'][node]
        self.Z = grid.at_node['catchment_average_regolith__thickness'][node]
        self.Zo = grid.at_node['regolith__thickness'][node]
        self.Ksat = grid.at_node['hydraulic_conductivity'][node]
        self.k_macropores = grid.at_node['hydraulic_conductivity_f'][node]
        self.gradZo = np.max(grid.at_node['topographic__steepest_slope'][node])
        self.k = grid.at_node['linear_reservoir_constant'][node]
        self.n_manning = grid.at_node['manning_n'][node]

        self.ETmax = grid.at_node['Maximum_ET_rate'][node]
        self.sw = grid.at_node['soil_wilting_point'][node]
        self.sf = grid.at_node['soil_field_capacity'][node]

        self.vo = self.gradZo*self.Ksat*self.k_macropores
        self.St = 1*self.n*self.Z*self.A
        self.Sf = self.sf*self.n*self.Z*self.A
        self.Sw = self.sw*self.n*self.Z*self.A

        self.S = self.Sw
        self.Qs = 0
        self.Qrc = 0
        self.Qh0 = 0

    def run_one_step(self,duration,intensity):


        """" Output Variable Descriptions

        soil moisture and local recharge and saturation excess:
        S   Soil water storage [L^3]
        f   infiltration rate [L/T]
        Qh  hortonian overland flow rate [L^3/T]
        Qr  Leakage rate [L^3/T]
        Qs  surface runoff rate [L^3/T]
        Qet evaporation rate [L^3/T]

        catchment scale recharge and subsurface flow and saturation excess overland flow
        Qrc catchment runoff from linear reservoir [L^3/T]
        Qsc catchment surface runoff [L^3/T]
        Qbc catchment subsurface runoff[L^3/T]


        """

        self.duration=duration
        self.intensity=intensity

        dt=duration
        Ksat = self.Ksat
        A = self.A
        Sw = self.Sw
        St = self.St
        Sf = self.Sf
        ETmax = self.ETmax
        vo = self.vo
        Zo = self.Zo
        gradZo = self.gradZo
        w = self.w
        k = self.k
        n = self.n
        n_manning = self.n_manning

        # infiltration and infiltration excess for timestep
        self.f = min(intensity, Ksat)
        self.Qh = max(0,intensity - Ksat)*A

        # soil moisture losses for timestep
        self.Qet = EvapotranspirationLoss(intensity, self.S, Sw,
                                          St, ETmax, A, dt)
        self.Qr = LeakageLoss(self.S,self.f,self.Qet,self.Qh0,self.Qs,
                              self.Qrc,dt,Sf,St,Ksat,A,vo,Zo,w)
        self.Qs = SaturationExcessLoss(self.S,self.Qr,self.f,St,Sf,A,dt)

        # subsurface and surface flow
        self.Qbc = min(vo*Zo*w, self.Qrc + self.Qs + self.Qh)
        self.Qsc = max((self.Qrc + self.Qs + self.Qh - vo*Zo*w),0)

        # update soil moisture and linear reservoir
        self.S = self.S -dt*self.Qet - dt*self.Qr - dt*self.Qs + dt*self.f*A
        self.Qrc = (1-k*dt)*self.Qrc + k*dt*self.Qr

        # calculate shear stress
        self.Taub = self.rho*self.g*gradZo* (
                    ((n_manning*(self.Qsc/3600)/w)*(1/np.sqrt(gradZo)))**(3/5) )

        # update previous timestep infiltration excess (used for Leakage loss)
        self.Qh0 = self.Qh

    def yield_all_timestep_data(self):
        return [self.duration, self.intensity, self.f, self.Qh, self.Qet, self.Qr,
                self.Qs, self.S, self.Qrc, self.Qbc, self.Qsc, self.Taub]

    def yield_static_properties(self):
        return [self.A, self.w, self.n, self.Z, self.Zo, self.Ksat,
                self.k_macropores, self.gradZo, self.k, self.ETmax, self.sw, self.sf]
