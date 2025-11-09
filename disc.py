#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 18:20:58 2025

This file contains the three different disc models of plot 7.
ProtoplanetaryDisc_constZ() --- assumes constant St, Z0, chi
ProtoplanetaryDisc_Zt()     --- assumes constant St*chi
ProtoplanetaryDisc_Zrt()    --- assumes constant St

@author: nerea
"""

import astropy.units as u
import numpy as np
from astropy.constants import G



# Disc with constant Z


class ProtoplanetaryDisc_constZ():
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7):
        self.t0 = t0                # initial time of the disc
        self.Mstar = Mstar          # mass of the star
        self.M_dot_0 = Mdot         # initial gas accretion rate towards star
        self.Z0 = Z0                # initial dust to gas ratio
        self.alpha = alpha          # accretion coefficient
        self.St_ = St               # characteristic Stokes number
        self.delta = delta          # turbulence coefficient
        self.cs1 = cs1              # sound speed at 1 AU
        self.zeta = zeta            # power index of temperature
        self.gamma = 3/2 - self.zeta # power index of gas surface density
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta)  # constant for migration rate
        self.chi0 = self.gamma + self.zeta/2 + 3/2 # -logP/logr at inner disc
        self.__R1 = R1              # disc size
        self.tvis = self.calculate_tvis(R1)  # viscous timescale
        
    
    # to calculate viscous timescale
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    # this model assumes constant Stokes number
    def St(self, r):
        return self.St_

    # gas surface density
    def sigma_g(self, r, t, tunit):
        total = (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        total *= T**power
        total *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return total
    
    # gas scale height
    def H(self, r):
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)

    # sound speed
    def cs(self, r):
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    # viscosity
    def visc(self, r):
        return self.alpha*self.cs(r)*self.H(r)

    # angular frequency
    def kepler_angular(self, r):
        return (G*self.Mstar/r**3)**(1/2)
    
    # pebble surface density
    def sigma_p(self, r, t, tunit):
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)

    # mass of the disc mass
    def disc_mass(self, t, tunit):
        import scipy.integrate        
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.0001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass) 
    
    # to change disc size; whenever we do so, viscous timescale needs to change
    def set_R1(self, R1):
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)  
    
    # get disc size
    def get_R1(self):
        return self.__R1   
    
    # radial velocity of the gas
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)

    # radial velocity of solids
    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r))/(1 + self.St(r)**2)).to(u.m/u.s)
        
    # gas flux
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

    # pebble flux
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               
    # keplerian reduction
    def delta_v(self, r, t):
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
    
    
    # the negative logarithmic pressure gradient in the  midplane
    # for this model, we assume that it is constant
    def chi(self, r, t):
        return self.chi0
    
    # dust-to-gas ratio
    def Z(self, r, t):
        return self.Z0




# Disc with Z(t), constant St*chi


class ProtoplanetaryDisc_Zt():
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7):
        self.t0 = t0                # initial time of the disc
        self.Mstar = Mstar          # mass of the star
        self.M_dot_0 = Mdot         # initial gas accretion rate towards star
        self.Z0 = Z0                # initial dust to gas ratio
        self.alpha = alpha          # accretion coefficient
        self.St_ = St               # characteristic Stokes number
        self.delta = delta          # turbulence coefficient
        self.cs1 = cs1              # sound speed at 1 AU
        self.zeta = zeta            # power index of temperature
        self.gamma = 3/2 - self.zeta # power index of gas surface density
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta)  # constant for migration rate
        self.chi0 = self.gamma + self.zeta/2 + 3/2 # -logP/logr at inner disc
        self.__R1 = R1              # disc size
        self.tvis = self.calculate_tvis(R1)  # viscous timescale
        self.b0 = self.b0_calculate()        # parameter that goes of the exponent of Z(t)   
        
    
    # to calculate viscous timescale
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    # this model assumes constant Stokes number
    def St(self, r, t):
        return self.b0*3*self.alpha/2/self.chi(r, t)

    # gas surface density
    def sigma_g(self, r, t, tunit):
        total = (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        total *= T**power
        total *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return total
    
    # gas scale height
    def H(self, r):
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)

    # sound speed
    def cs(self, r):
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    # viscosity
    def visc(self, r):
        return self.alpha*self.cs(r)*self.H(r)

    # angular frequency
    def kepler_angular(self, r):
        return (G*self.Mstar/r**3)**(1/2)
    
    # pebble surface density
    def sigma_p(self, r, t, tunit):
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)

    # mass of the disc mass
    def disc_mass(self, t, tunit):
        import scipy.integrate        
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.0001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass) 
    
    # to change disc size; whenever we do so, viscous timescale needs to change
    def set_R1(self, R1):
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)  
    
    # get disc size
    def get_R1(self):
        return self.__R1   
    
    # radial velocity of the gas
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)

    # radial velocity of solids
    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r,t))/(1 + self.St(r,t)**2)).to(u.m/u.s)
        
    # gas flux
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

    # pebble flux
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               
    # keplerian reduction
    def delta_v(self, r, t):
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
    
    
    # the negative logarithmic pressure gradient in the  midplane
    # for this model, we assume that it is constant
    def chi(self, r, t):
        T = 1 + (t-self.t0)/self.tvis
        return self.chi0 + (2-self.gamma)*(r/self.get_R1())**(2-self.gamma)/T
    
    # dust-to-gas ratio
    def Z(self, r, t):
        Z = self.Z0*(1 + (t-self.t0)/self.tvis)**(-self.b0/(2-self.gamma)/2)
        return Z.decompose()
    
    # to compute b0 parameter, which goes on the exponent of Z
    def b0_calculate(self):
        return 2/3*self.chi(self.__R1, self.t0)*self.St_/self.alpha




# Disc with Z(r, t), constant St


class ProtoplanetaryDisc_Zrt():
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7):
        self.t0 = t0                # initial time of the disc
        self.Mstar = Mstar          # mass of the star
        self.M_dot_0 = Mdot         # initial gas accretion rate towards star
        self.Z0 = Z0                # initial dust to gas ratio
        self.alpha = alpha          # accretion coefficient
        self.St_ = St               # characteristic Stokes number
        self.delta = delta          # turbulence coefficient
        self.cs1 = cs1              # sound speed at 1 AU
        self.zeta = zeta            # power index of temperature
        self.gamma = 3/2 - self.zeta # power index of gas surface density
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta)  # constant for migration rate
        self.chi0 = self.gamma + self.zeta/2 + 3/2 # -logP/logr at inner disc
        self.__R1 = R1              # disc size
        self.tvis = self.calculate_tvis(R1)  # viscous timescale
        self.b0 = self.b0_calculate()        # parameter that goes of the exponent of Z(t)   
        
    
    # to calculate viscous timescale
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    # this model assumes constant Stokes number
    def St(self, r, t):
        return self.St_

    # gas surface density
    def sigma_g(self, r, t, tunit):
        total = (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        total *= T**power
        total *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return total
    
    # gas scale height
    def H(self, r):
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)

    # sound speed
    def cs(self, r):
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    # viscosity
    def visc(self, r):
        return self.alpha*self.cs(r)*self.H(r)

    # angular frequency
    def kepler_angular(self, r):
        return (G*self.Mstar/r**3)**(1/2)
    
    # pebble surface density
    def sigma_p(self, r, t, tunit):
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)

    # mass of the disc mass
    def disc_mass(self, t, tunit):
        import scipy.integrate        
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.0001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass) 
    
    # to change disc size; whenever we do so, viscous timescale needs to change
    def set_R1(self, R1):
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)  
    
    # get disc size
    def get_R1(self):
        return self.__R1   
    
    # radial velocity of the gas
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)

    # radial velocity of solids
    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r,t))/(1 + self.St(r,t)**2)).to(u.m/u.s)
        
    # gas flux
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

    # pebble flux
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               
    # keplerian reduction
    def delta_v(self, r, t):
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
    
    
    # the negative logarithmic pressure gradient in the  midplane
    # for this model, we assume that it is constant
    def chi(self, r, t):
        T = 1 + (t-self.t0)/self.tvis
        return self.chi0 + (2-self.gamma)*(r/self.get_R1())**(2-self.gamma)/T


    # dust-to-gas ratio
    def Z(self, r, t):
        r_nu = (r/self.get_R1())
        T = 1 + (t-self.t0)/self.tvis
        exp_ = np.exp(-(self.chi0*(self.b0+1)/(2-self.gamma)*T + 
                        r_nu**(2-self.gamma)*self.b0)*(T**(self.b0/2/self.chi0)-1)/(self.b0*T))
        Z=self.Z0*T**((self.chi0/(2-self.gamma) + self.b0)/2/self.chi0)*exp_
        return Z.decompose()#(self.xi*np.exp((1-vr_ur)*(t-self.t0)/self.lifetime)).decompose()

    
    # to compute b0 parameter, which goes on the exponent of Z; note that 
    # chi is slightly different to b0 from ProtoplanetaryDisc_Zt
    def b0_calculate(self):
        return 2/3*self.chi0*self.St_/self.alpha

