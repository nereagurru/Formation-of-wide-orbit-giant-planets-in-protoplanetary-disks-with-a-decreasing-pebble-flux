#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:18:20 2023

@author: nerea
"""


import astropy.units as u
import numpy as np
from astropy.constants import G


# A code that gathers all the different disks

# In this code St constant by default


# ---- First disk, constant pebble flux from Johansen et al. 2019 -------

class ProtoplanetaryDisc():
    
    gasDisipation = True # The gas in the disc decreases with time
    exponential = True # The profile of the gas density ends with an exponential
    St_cons = True # If true, Stokes number constant (and equal to alpha)
    outwards_flux = False
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s,
                 zeta=3./7, vf=1*u.m/u.s, constant_flux=True, last_pebble_pos_init=300*u.AU):
        """
        Initialise protoplanetary disc.
    
        Parameters
        ----------

        t0 : float, time dim
            Time at which pebble accretion start to be significant.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        xi : float, dimless
            The ratio of fluxes, defined in eq. 15.
        R1 : float, lenght dim
            The initial characteristic size of the gas disk.
        Mstar : float, mass dim
            The mass pf the embryon star
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        """
        self.t0 = t0
        self.Mstar = Mstar
        self.M_dot_0 = Mdot #self.calculate_M_dot_0()
        self.alpha = alpha
        self.St_ = St
        self.Z0 = Z0
        
        self.delta = delta
        self.vf = vf
        self.cs1 = cs1
        self.zeta = zeta # the negative power-law index of the temperature
        self.gamma = 3/2 - self.zeta # the negative power-law index of the surface
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta) 
        self.chi0 = self.gamma + self.zeta/2 + 3/2
        
        self.xi = Z0*(1 + (2/3)*(St/alpha)*self.chi0)/(1 + St**2)  # Be careful with this, it is for constant St
        self.__R1 = R1 # Be careful with changing R1, we can 
        self.tvis = self.calculate_tvis(R1)
        self.last_pebble_pos_init = last_pebble_pos_init
        self.constant_flux = constant_flux
        if self.constant_flux == False:
            self.last_pebble_pos_arr, self.last_pebble_time_arr = self.flux_pos_fill()
        else:
            self.last_pebble_pos_arr = None
            self.last_pebble_time_arr = None
        
        
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    
    def St(self, r, t):
        """
        To calculate Stokes number at certain distance, defined as in 
        (Johansen and Lambrechts, 2017)

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float
            The Stokes number (St)

        """
        if self.St_cons:
            return self.St_
        else:
            return (((self.vf/self.cs(r))**2)/3/self.delta).decompose()


    def calculate_M_dot_0(self):
        """
        To calculate the initial mass accretion rate, from observations in Liu 2020.
        
        Parameters
        ----------

        Returns
        -------
        float, Mass per time units
            The mass accretion rate at t0

        """
        """
        """
        M_dot = 10**-7
        return M_dot*u.solMass/u.yr
    
    
    
    def choose_model(self, gasDis=True, exponent=True, Stcons=True, Mdot=False, R1calculate=False, outwards=False):
        self.gasDisipation = gasDis
        self.exponential = exponent
        self.St_cons = Stcons
        self.outwards_flux = outwards
        if Mdot:
            self.M_dot_0 = 10**(-7)*u.solMass/u.yr
        if R1calculate:
            self.__R1 = self.calculate_R1()
            self.tvis = self.calculate_tvis(self.__R1)


    def initial_gas_function(self, r):
        """
        To calculate the local initial value of the gas density surface, defined as in 
        Ida 2016 and Hartmann 1998.

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float, dim of Mass/Length**2
            The initial gas density at r distance from the star.

        """

        return (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
    
        
    def sigma_g(self, r, t, tunit):
        """ 
        To calculate the local value of the gas density surface at a certain time.
        Only gas accretion onto the star is taken into account. (No photoevaporation)
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the local gas density surface.
        t : float
            Time to evaluate the gas density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        

        Returns
        -------
        float, dim of Mass/Length**2
            Local surface density of the gas at a certaint time.

        """
    
        # We assume that the there is only one possible solution for the density 
        result = self.initial_gas_function(r)
        if self.gasDisipation:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            power = -(5/2-self.gamma)/(2-self.gamma)
            result *= T**power
        else:
            T = 1    
        if self.exponential:
            result *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return result
    
    
    def H(self, r):
        # As the disc is vertically isothermal
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)


    def cs(self, r):
        """ 
        To calculate the local speed of sound. Eq.5 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the speed.

        Returns
        -------
        float, dim of Length/Time
            Local sound speed of the gas at a certain distance.

        """
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    
    def visc(self, r):
        """ 
        To calculate the local viscosity. Eq.7 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the viscosity.

        Returns
        -------
        float, dim of Length**2/Time
            The viscosity of the gas at a certain distance.

        """
        return self.alpha*self.cs(r)*self.H(r)

    
    def kepler_angular(self, r):
        """ 
        To calculate the Keplerian angular velocity.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the angular velocity.

        Returns
        -------
        float, dim of 1/Time
            Keplerian velocity at certain distance.

        """
        return (G*self.Mstar/r**3)**(1/2)
    
    
    def sigma_p(self, r, t, tunit):
        """ 
        To calculate the pebble surface density.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the density.
        t : float
            Time to evaluate the pebble density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        
        Returns
        -------
        float, dim of mass/length**2
            Surface density of pebbles at a certain distance and time.

        """
        return self.sigma_g(r, t, tunit)*self.Z0

        
    
    def disc_mass(self, t, tunit):
        """
        Function to calculate the mass of the disk at certain time.
        Right now it depends on R1.

        Parameters
        ----------
        t : float
            Time at which the disk mass is evaluated.
        tunit : unit of time
            Timescale of the time

        Returns
        -------
        float, mass units 
            mass of the disk
            
        """
        import scipy.integrate        
        def f(r, t, tunit):
            return r*self.sigma_g(r*u.AU, t, tunit).value
        
        # The integral has an error that we are not taking into account. to do it check result[1] instead of result[0]
        return (2*np.pi*scipy.integrate.quad(f, 0, np.inf, args=(t, tunit))[0]*(u.AU**2)*self.sigma_g(1*u.AU, t, tunit).unit).to(u.solMass)
    
    
    def f_root(self, R1):
        # From 10**-7 to 10**-8 during 3 Myr
        R1 = R1*u.AU
        power = -(5/2 - self.gamma)/(2-self.gamma)
        Mt = 10**(-8)*u.Msun/u.yr
        M0 = 10**(-7)*u.Msun/u.yr
        t = 3*u.Myr
        tvis = self.calculate_tvis(R1)
        return Mt - M0*(t/tvis + 1)**power
        # return 10**(np.log10(0.1)/power) - (3*u.Myr/self.calculate_tvis(R1)).decompose() - 1
    
    
    def calculate_R1(self):
        import scipy.optimize
        R1 = scipy.optimize.fsolve(self.f_root, 1)
        return R1[0]*u.AU

    def set_R1(self, R1):
        # I create this function because whenever R1 is changed, tvis should be changed.
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)    
    
    
    def get_R1(self):
        return self.__R1
    
    
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)

    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r))/(1 + self.St(r)**2)).to(u.m/u.s)
        
    
    def non_analytical_flux(self, r):
        vr = -(self.chi0*self.St(r) + (3/2)*self.alpha)/(self.St(r)**2 + 1)*self.cs1**2*(u.AU/G/u.Msun)**(1/2)*(r/u.AU)**(-self.zeta +1/2)
        return 1/vr


    def flux_pos_fill(self):
        deltat = 0.001
        unit = u.Myr
        t = np.arange(self.t0.value, 3, deltat)*unit
        r = np.empty(t.shape)*u.AU
        r[0] = self.last_pebble_pos_init
        for i in range(1, t.shape[0]):
            if r[i-1] < 0.5*u.AU:
                r[i:] = 0*u.AU
                break
            else:
                r[i] = self.vr_solid(r[i-1], t[i-1])*(deltat*unit) + r[i-1]

        return r, t.to(u.Myr)


    def last_pebble_pos(self, t):
        return np.interp(t, self.last_pebble_time_arr, self.last_pebble_pos_arr)


    def last_pebble_time(self, r):
        return np.interp(r, np.flip(self.last_pebble_pos_arr), np.flip(self.last_pebble_time_arr))

    

    def Mg_dot(self, r, t, tunit):
        result = self.M_dot_0.copy()
        if self.gasDisipation:
            power = -(5/2-self.gamma)/(2-self.gamma)
            T = (1 + (t*tunit-self.t0)/self.tvis)
            result *= T**power
        if self.outwards_flux:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            result *=np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))
        return result


    def Mp_dot(self, r, t, tunit):
        return np.abs(self.Mg_dot(r, t, tunit)*self.xi)
           

    def delta_v(self, r, t):
        # SubKeplerian Velocity
        return 0.5*(self.H(r)/r)*self.chi0*self.cs(r)

















#------------------- Disk with Sigmap = Z0*Sigmag ------------------


class ProtoplanetaryDisc1():
    
    gasDisipation = True # The gas in the disc decreases with time
    exponential = True # The profile of the gas density ends with an exponential
    St_cons = True # If true, Stokes number constant (and equal to alpha)
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7,
                 vf=1*u.m/u.s, constant_flux=True, last_pebble_pos_init=300*u.AU):
        """
        Initialise protoplanetary disc.
    
        Parameters
        ----------

        t0 : float, time dim
            Time at which pebble accretion start to be significant.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        xi : float, dimless
            The ratio of fluxes, defined in eq. 15.
        R1 : float, lenght dim
            The initial characteristic size of the gas disk.
        Mstar : float, mass dim
            The mass pf the embryon star
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        """
        self.t0 = t0
        self.Mstar = Mstar
        self.M_dot_0 = Mdot #self.calculate_M_dot_0()
        self.Z0 = Z0
        self.alpha = alpha
        self.St_ = St
        self.delta = delta
        self.vf = vf
        self.cs1 = cs1
        self.zeta = zeta # the negative power-law index of the temperature
        self.gamma = 3/2 - self.zeta # the negative power-law index of the surface
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta) 
        self.chi0 = self.gamma + self.zeta/2 + 3/2
        
        self.xi = Z0*(1 + (2/3)*(St/alpha)*self.chi0)/(1 + St**2) # Be careful with this, it is for constant St
  
        self.__R1 = R1 # Be careful with changing R1, we can 
        self.tvis = self.calculate_tvis(R1)
        self.last_pebble_pos_init = last_pebble_pos_init
        self.constant_flux = constant_flux
        if self.constant_flux == False:
            self.last_pebble_pos_arr, self.last_pebble_time_arr = self.flux_pos_fill()
        else:
            self.last_pebble_pos_arr = None
            self.last_pebble_time_arr = None
        
        
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3./(2.-self.gamma)**2).to(u.Myr)
    
    
    def St(self, r):
        """
        To calculate Stokes number at certain distance, defined as in 
        (Johansen and Lambrechts, 2017)

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float
            The Stokes number (St)

        """
        if self.St_cons:
            return self.St_
        else:
            return (((self.vf/self.cs(r))**2)/3/self.delta).decompose()


    def calculate_M_dot_0(self):
        """
        To calculate the initial mass accretion rate, from observations in Liu 2020.
        
        Parameters
        ----------

        Returns
        -------
        float, Mass per time units
            The mass accretion rate at t0

        """
        """
        if self.t0 >= 0.3*u.Myr:
            t = self.t0
        else:
            t = 0.3*u.Myr
            
        M_dot = 10**(-5.12 - 0.46*np.log10(t/u.yr) 
                     - 5.75*np.log10(self.Mstar/u.solMass)
                     + 1.17*np.log10(t/u.yr)*np.log10(self.Mstar/u.solMass))
        """
        M_dot = 10**-7
        return M_dot*u.solMass/u.yr
    
    
    
    def choose_model(self, gasDis=True, exponent=True, Stcons=False, Mdot=False, R1calculate=False):
        self.gasDisipation = gasDis
        self.exponential = exponent
        self.St_cons = Stcons
        if Mdot:
            self.M_dot_0 = 10**(-7)*u.solMass/u.yr
        if R1calculate:
            self.__R1 = self.calculate_R1()
            self.tvis = self.calculate_tvis(self.__R1)


    def initial_gas_function(self, r):
        """
        To calculate the local initial value of the gas density surface, defined as in 
        Ida 2016 and Hartmann 1998.

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float, dim of Mass/Length**2
            The initial gas density at r distance from the star.

        """

        return (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
    
        
    def sigma_g(self, r, t, tunit):
        """ 
        To calculate the local value of the gas density surface at a certain time.
        Only gas accretion onto the star is taken into account. (No photoevaporation)
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the local gas density surface.
        t : float
            Time to evaluate the gas density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        

        Returns
        -------
        float, dim of Mass/Length**2
            Local surface density of the gas at a certaint time.

        """
    
        # We assume that the there is only one possible solution for the density 
        result = self.initial_gas_function(r)
        if self.gasDisipation:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            power = -(5/2-self.gamma)/(2-self.gamma)
            result *= T**power
        else:
            T = 1    
        if self.exponential:
            result *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return result
    
    
    def H(self, r):
        # As the disc is vertically isothermal
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)


    def cs(self, r):
        """ 
        To calculate the local speed of sound. Eq.5 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the speed.

        Returns
        -------
        float, dim of Length/Time
            Local sound speed of the gas at a certain distance.

        """
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    
    def visc(self, r):
        """ 
        To calculate the local viscosity. Eq.7 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the viscosity.

        Returns
        -------
        float, dim of Length**2/Time
            The viscosity of the gas at a certain distance.

        """
        return self.alpha*self.cs(r)*self.H(r)

    
    def kepler_angular(self, r):
        """ 
        To calculate the Keplerian angular velocity.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the angular velocity.

        Returns
        -------
        float, dim of 1/Time
            Keplerian velocity at certain distance.

        """
        return (G*self.Mstar/r**3)**(1/2)
    
    
    def sigma_p(self, r, t, tunit):
        """ 
        To calculate the pebble surface density.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the density.
        t : float
            Time to evaluate the pebble density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        
        Returns
        -------
        float, dim of mass/length**2
            Surface density of pebbles at a certain distance and time.

        """
        return self.sigma_g(r, t, tunit)*self.Z0


        
    
    def disc_mass(self, t, tunit):
        """
        Function to calculate the mass of the disk at certain time.
        Right now it depends on R1.

        Parameters
        ----------
        t : float
            Time at which the disk mass is evaluated.
        tunit : unit of time
            Timescale of the time

        Returns
        -------
        float, mass units 
            mass of the disk
            
        """
        import scipy.integrate        
        def f(r, t, tunit):
            return r*self.sigma_g(r*u.AU, t, tunit).value
        
        # The integral has an error that we are not taking into account. to do it check result[1] instead of result[0]
        return (2*np.pi*scipy.integrate.quad(f, 0, np.inf, args=(t, tunit))[0]*(u.AU**2)*self.sigma_g(1*u.AU, t, tunit).unit).to(u.solMass)
    
    
    def f_root(self, R1):
        # From 10**-7 to 10**-8 during 3 Myr
        R1 = R1*u.AU
        power = -(5/2 - self.gamma)/(2-self.gamma)
        Mt = 10**(-8)*u.Msun/u.yr
        M0 = 10**(-7)*u.Msun/u.yr
        t = 3*u.Myr
        tvis = self.calculate_tvis(R1)
        return Mt - M0*(t/tvis + 1)**power
        # return 10**(np.log10(0.1)/power) - (3*u.Myr/self.calculate_tvis(R1)).decompose() - 1
    
    
    def calculate_R1(self):
        import scipy.optimize
        R1 = scipy.optimize.fsolve(self.f_root, 1)
        return R1[0]*u.AU

    def set_R1(self, R1):
        # I create this function because whenever R1 is changed, tvis should be changed.
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)    
    
    
    def get_R1(self):
        return self.__R1
        

    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)


    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r))/(1 + self.St(r)**2)).to(u.m/u.s)

        
    def non_analytical_flux(self, r):
        vr = -(self.chi0*self.St(r) + (3/2)*self.alpha)/(self.St(r)**2 + 1)*self.cs1**2*(u.AU/G/u.Msun)**(1/2)*(r/u.AU)**(-self.zeta +1/2)
        return 1/vr


    def flux_pos_fill(self):
        deltar = 0.01
        unit = self.last_pebble_pos_init.unit
        r = np.arange(deltar, self.last_pebble_pos_init.value + deltar, deltar)*unit
        r = np.flip(r)
        t = np.cumsum(self.non_analytical_flux(r))*(-deltar*unit) + self.t0
        return r, t.to(u.Myr)


    def last_pebble_pos(self, t):
        return np.interp(t, self.last_pebble_time_arr, self.last_pebble_pos_arr)


    def last_pebble_time(self, r):
        return np.interp(r, np.flip(self.last_pebble_pos_arr), np.flip(self.last_pebble_time_arr))
    
    
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))
        
    
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z0*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
    
    
    def delta_v(self, r, t):
        # SubKeplerian Velocity
        return 0.5*(self.H(r)/r)*self.chi0*self.cs(r)








#------------------- Disk with Sigmap = Z(t)*Sigmag ------------------

class ProtoplanetaryDisc2():
    
    gasDisipation = True # The gas in the disc decreases with time
    exponential = True # The profile of the gas density ends with an exponential
    St_cons = True # If true, Stokes number constant 
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7, vf=1*u.m/u.s, lifetime=3*u.Myr):
        """
        Initialise protoplanetary disc.
    
        Parameters
        ----------

        t0 : float, time dim
            Time at which pebble accretion start to be significant.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        xi : float, dimless
            The ratio of fluxes, defined in eq. 15.
        R1 : float, lenght dim
            The initial characteristic size of the gas disk.
        Mstar : float, mass dim
            The mass pf the embryon star
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        """
        self.t0 = t0
        self.lifetime = lifetime
        self.Mstar = Mstar
        self.M_dot_0 = Mdot #self.calculate_M_dot_0()
        self.Z0 = Z0
        self.alpha = alpha
        self.St_ = St
        self.delta = delta
        self.vf = vf
        self.cs1 = cs1
        self.zeta = zeta # the negative power-law index of the temperature
        self.gamma = 3/2 - self.zeta # the negative power-law index of the surface
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta) 
        self.chi0 = self.gamma + self.zeta/2 + 3/2
        
        self.xi = Z0*(1 + (2/3)*(St/alpha)*self.chi0)/(1 + St**2) 
        self.__R1 = R1 # Be careful with changing R1, we can 
        self.tvis = self.calculate_tvis(R1)
        self.constant_flux = True
        self.b0 = self.b0_calculate()
        
        
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3./(2.-self.gamma)**2).to(u.Myr)
    
    
    def St(self, r, t):
        """
        To calculate Stokes number at certain distance, defined as in 
        (Johansen and Lambrechts, 2017)

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float
            The Stokes number (St)

        """
        return self.b0*3*self.alpha/2/self.chi(r, t)


    def chi(self, r, t):
        T = 1 + (t-self.t0)/self.tvis
        return self.chi0 + (2-self.gamma)*(r/self.get_R1())**(2-self.gamma)/T
    
    
    def b0_calculate(self):
        return 2/3*self.chi(self.__R1, self.t0)*self.St_/self.alpha
    
    
    def calculate_M_dot_0(self):
        """
        To calculate the initial mass accretion rate, from observations in Liu 2020.
        
        Parameters
        ----------

        Returns
        -------
        float, Mass per time units
            The mass accretion rate at t0

        """
        """
        if self.t0 >= 0.3*u.Myr:
            t = self.t0
        else:
            t = 0.3*u.Myr
            
        M_dot = 10**(-5.12 - 0.46*np.log10(t/u.yr) 
                     - 5.75*np.log10(self.Mstar/u.solMass)
                     + 1.17*np.log10(t/u.yr)*np.log10(self.Mstar/u.solMass))
        """
        M_dot = 10**-7
        return M_dot*u.solMass/u.yr
    
    
    
    def choose_model(self, gasDis=True, exponent=True, Stcons=True, Mdot=False, R1calculate=False):
        self.gasDisipation = gasDis
        self.exponential = exponent
        self.St_cons = Stcons
        if Mdot:
            self.M_dot_0 = 10**(-7)*u.solMass/u.yr
        if R1calculate:
            self.__R1 = self.calculate_R1()
            self.tvis = self.calculate_tvis(self.__R1)


    def initial_gas_function(self, r):
        """
        To calculate the local initial value of the gas density surface, defined as in 
        Ida 2016 and Hartmann 1998.

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float, dim of Mass/Length**2
            The initial gas density at r distance from the star.

        """

        return (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
    
        
    def sigma_g(self, r, t, tunit):
        """ 
        To calculate the local value of the gas density surface at a certain time.
        Only gas accretion onto the star is taken into account. (No photoevaporation)
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the local gas density surface.
        t : float
            Time to evaluate the gas density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        

        Returns
        -------
        float, dim of Mass/Length**2
            Local surface density of the gas at a certaint time.

        """
    
        # We assume that the there is only one possible solution for the density 
        result = self.initial_gas_function(r)
        if self.gasDisipation:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            power = -(5/2-self.gamma)/(2-self.gamma)
            result *= T**power
        else:
            T = 1    
        if self.exponential:
            result *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return result
    
    
    def H(self, r):
        # As the disc is vertically isothermal
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)


    def cs(self, r):
        """ 
        To calculate the local speed of sound. Eq.5 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the speed.

        Returns
        -------
        float, dim of Length/Time
            Local sound speed of the gas at a certain distance.

        """
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    
    def visc(self, r):
        """ 
        To calculate the local viscosity. Eq.7 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the viscosity.

        Returns
        -------
        float, dim of Length**2/Time
            The viscosity of the gas at a certain distance.

        """
        return self.alpha*self.cs(r)*self.H(r)

    
    def kepler_angular(self, r):
        """ 
        To calculate the Keplerian angular velocity.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the angular velocity.

        Returns
        -------
        float, dim of 1/Time
            Keplerian velocity at certain distance.

        """
        return (G*self.Mstar/r**3)**(1/2)
    
    
    def sigma_p(self, r, t, tunit):
        """ 
        To calculate the pebble surface density.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the density.
        t : float
            Time to evaluate the pebble density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        
        Returns
        -------
        float, dim of mass/length**2
            Surface density of pebbles at a certain distance and time.

        """
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)

    # Gives the same result as the one bellow
    #def Z(self, r, t):
    #    vr_ur = (1 + (2/3)*self.St(r)/self.alpha*self.chi0)/(1 + self.St(r)**2)
    #    Z = self.Z0*(1 + (t-self.t0)/self.tvis)**(-1/2/(2-self.gamma)*(vr_ur-1))
    #    return Z.decompose()#(self.xi*np.exp((1-vr_ur)*(t-self.t0)/self.lifetime)).decompose()
        
    
    def Z(self, r, t):
        Z = self.Z0*(1 + (t-self.t0)/self.tvis)**(-self.b0/(2-self.gamma)/2)
        return Z.decompose()#(self.xi*np.exp((1-vr_ur)*(t-self.t0)/self.lifetime)).decompose()
    
    
    def disc_mass(self, t, tunit):
        """
        Function to calculate the mass of the disk at certain time.
        Right now it depends on R1.

        Parameters
        ----------
        t : float
            Time at which the disk mass is evaluated.
        tunit : unit of time
            Timescale of the time

        Returns
        -------
        float, mass units 
            mass of the disk
            
        """
        import scipy.integrate        
        
        
        # The integral has an error that we are not taking into account. to do it check result[1] instead of result[0]
        #return (2*np.pi*scipy.integrate.quad(f, 0, np.inf, args=(t, tunit))[0]*(u.AU**2)*self.sigma_g(1*u.AU, t, tunit).unit).to(u.solMass)
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass)

    
    def f_root(self, R1):
        # From 10**-7 to 10**-8 during 3 Myr
        R1 = R1*u.AU
        power = -(5/2 - self.gamma)/(2-self.gamma)
        Mt = 10**(-8)*u.Msun/u.yr
        M0 = 10**(-7)*u.Msun/u.yr
        t = 3*u.Myr
        tvis = self.calculate_tvis(R1)
        return Mt - M0*(t/tvis + 1)**power
        # return 10**(np.log10(0.1)/power) - (3*u.Myr/self.calculate_tvis(R1)).decompose() - 1
    
    
    def calculate_R1(self):
        import scipy.optimize
        R1 = scipy.optimize.fsolve(self.f_root, 1)
        return R1[0]*u.AU


    def set_R1(self, R1):
        # I create this function because whenever R1 is changed, tvis should be changed.
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)    
    
    
    def get_R1(self):
        return self.__R1   
    
    
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)

    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r, t))).to(u.m/u.s)
        
    
    def non_analytical_flux(self, r):
        vr = -(self.chi0*self.St(r) + (3/2)*self.alpha)/(self.St(r)**2 + 1)*self.cs1**2*(u.AU/G/u.Msun)**(1/2)*(r/u.AU)**(-self.zeta +1/2)
        return 1/vr


    def flux_pos_fill(self):
        deltat = 0.001
        unit = u.Myr
        t = np.arange(self.t0.value, 3, deltat)*unit
        r = np.empty(t.shape)*u.AU
        r[0] = self.last_pebble_pos_init
        for i in range(1, t.shape[0]):
            if r[i-1] < 0.5*u.AU:
                r[i:] = 0*u.AU
                break
            else:
                r[i] = self.vr_solid(r[i-1], t[i-1])*(deltat*unit) + r[i-1]

        return r, t.to(u.Myr)


    def last_pebble_pos(self, t):
        return np.interp(t, self.last_pebble_time_arr, self.last_pebble_pos_arr)


    def last_pebble_time(self, r):
        return np.interp(r, np.flip(self.last_pebble_pos_arr), np.flip(self.last_pebble_time_arr))


    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

        
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               

    def delta_v(self, r, t):
        # SubKeplerian Velocity
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
                      
                                                                
    def Mg(self, t):
        T = (1 + (t-self.t0)/self.tvis)
        power = -1/(2-self.gamma)/2
        return 2/3*self.M_dot_0/self.visc(self.__R1)*self.__R1**2/(2-self.gamma)*T**power
    
    
    
    
    
    
    
    
    

#------------------- Disk with Sigmap = Z(r, t)*Sigmag and non-constant chi ------------------

class ProtoplanetaryDisc3():
    
    gasDisipation = True # The gas in the disc decreases with time
    exponential = True # The profile of the gas density ends with an exponential
    St_cons = True # If true, Stokes number constant 
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7, vf=1*u.m/u.s, lifetime=3*u.Myr):
        """
        Initialise protoplanetary disc.
    
        Parameters
        ----------

        t0 : float, time dim
            Time at which pebble accretion start to be significant.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        Z0 : float, dimless
            The ratio of fluxes, defined in eq. 15.
        R1 : float, lenght dim
            The initial characteristic size of the gas disk.
        Mstar : float, mass dim
            The mass pf the embryon star
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        """
        self.t0 = t0
        self.lifetime = lifetime
        self.Mstar = Mstar
        self.M_dot_0 = Mdot #self.calculate_M_dot_0()
        self.Z0 = Z0
        self.constant_flux = True # to avoid problems with planetesimal class
        self.alpha = alpha
        self.St_ = St
        self.delta = delta
        self.vf = vf
        self.cs1 = cs1
        self.zeta = zeta # the negative power-law index of the temperature
        self.gamma = 3/2 - self.zeta # the negative power-law index of the surface
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta) 
        self.chi0 = self.gamma + self.zeta/2 + 3/2
        
        self.xi = Z0*(1 + (2/3)*(St/alpha)*self.chi0)/(1 + St**2) 
        self.__R1 = R1 # Be careful with changing R1, we can 
        self.tvis = self.calculate_tvis(R1)
        
        
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    
    def St(self, r, t):
        """
        To calculate Stokes number at certain distance, defined as in 
        (Johansen and Lambrechts, 2017)

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float
            The Stokes number (St)

        """
        if self.St_cons:
            return self.St_
        else:
            return (((self.vf/self.cs(r))**2)/3/self.delta).decompose()


    def calculate_M_dot_0(self):
        """
        To calculate the initial mass accretion rate, from observations in Liu 2020.
        
        Parameters
        ----------

        Returns
        -------
        float, Mass per time units
            The mass accretion rate at t0

        """
        """
        if self.t0 >= 0.3*u.Myr:
            t = self.t0
        else:
            t = 0.3*u.Myr
            
        M_dot = 10**(-5.12 - 0.46*np.log10(t/u.yr) 
                     - 5.75*np.log10(self.Mstar/u.solMass)
                     + 1.17*np.log10(t/u.yr)*np.log10(self.Mstar/u.solMass))
        """
        M_dot = 10**-7
        return M_dot*u.solMass/u.yr
    
    
    
    def choose_model(self, gasDis=True, exponent=True, Stcons=True, Mdot=False, R1calculate=False):
        self.gasDisipation = gasDis
        self.exponential = exponent
        self.St_cons = Stcons
        if Mdot:
            self.M_dot_0 = 10**(-7)*u.solMass/u.yr
        if R1calculate:
            self.__R1 = self.calculate_R1()
            self.tvis = self.calculate_tvis(self.__R1)


    def initial_gas_function(self, r):
        """
        To calculate the local initial value of the gas density surface, defined as in 
        Ida 2016 and Hartmann 1998.

        Parameters
        ----------
        r : float, lenght units
            Distance from the star.

        Returns
        -------
        float, dim of Mass/Length**2
            The initial gas density at r distance from the star.

        """

        return (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
    
        
    def sigma_g(self, r, t, tunit):
        """ 
        To calculate the local value of the gas density surface at a certain time.
        Only gas accretion onto the star is taken into account. (No photoevaporation)
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the local gas density surface.
        t : float
            Time to evaluate the gas density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        

        Returns
        -------
        float, dim of Mass/Length**2
            Local surface density of the gas at a certaint time.

        """
    
        # We assume that the there is only one possible solution for the density 
        result = self.initial_gas_function(r)
        if self.gasDisipation:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            power = -(5/2-self.gamma)/(2-self.gamma)
            result *= T**power
        else:
            T = 1    
        if self.exponential:
            result *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return result
    
    
    def H(self, r):
        # As the disc is vertically isothermal
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)


    def cs(self, r):
        """ 
        To calculate the local speed of sound. Eq.5 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the speed.

        Returns
        -------
        float, dim of Length/Time
            Local sound speed of the gas at a certain distance.

        """
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    
    def visc(self, r):
        """ 
        To calculate the local viscosity. Eq.7 from Johansen 2019
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the viscosity.

        Returns
        -------
        float, dim of Length**2/Time
            The viscosity of the gas at a certain distance.

        """
        return self.alpha*self.cs(r)*self.H(r)

    
    def kepler_angular(self, r):
        """ 
        To calculate the Keplerian angular velocity.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the angular velocity.

        Returns
        -------
        float, dim of 1/Time
            Keplerian velocity at certain distance.

        """
        return (G*self.Mstar/r**3)**(1/2)
    
    
    def sigma_p(self, r, t, tunit):
        """ 
        To calculate the pebble surface density.
        
        Parameters
        ----------
        r : float, dim of length
            Position to evaluate the density.
        t : float
            Time to evaluate the pebble density surface. If the initial time t0!=0,
            the given t must be t - t0.
        tunit: unit of time
        
        Returns
        -------
        float, dim of mass/length**2
            Surface density of pebbles at a certain distance and time.

        """
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)


    def Z(self, r, t):
        b0 =  2*self.St_/3/self.alpha*self.chi0
        r_nu = (r/self.get_R1())
        T = 1 + (t-self.t0)/self.tvis
        exp_ = np.exp(-(self.chi0*(b0+1)/(2-self.gamma)*T + 
                        r_nu**(2-self.gamma)*b0)*(T**(b0/2/self.chi0)-1)/(b0*T))
        Z=self.Z0*T**((self.chi0/(2-self.gamma) + b0)/2/self.chi0)*exp_
        return Z.decompose()#(self.xi*np.exp((1-vr_ur)*(t-self.t0)/self.lifetime)).decompose()
        
    
    def disc_mass(self, t, tunit):
        """
        Function to calculate the mass of the disk at certain time.
        Right now it depends on R1.

        Parameters
        ----------
        t : float
            Time at which the disk mass is evaluated.
        tunit : unit of time
            Timescale of the time

        Returns
        -------
        float, mass units 
            mass of the disk
            
        """
        import scipy.integrate        
        
        
        # The integral has an error that we are not taking into account. to do it check result[1] instead of result[0]
        #return (2*np.pi*scipy.integrate.quad(f, 0, np.inf, args=(t, tunit))[0]*(u.AU**2)*self.sigma_g(1*u.AU, t, tunit).unit).to(u.solMass)
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass)


    
    
    def f_root(self, R1):
        # From 10**-7 to 10**-8 during 3 Myr
        R1 = R1*u.AU
        power = -(5/2 - self.gamma)/(2-self.gamma)
        Mt = 10**(-8)*u.Msun/u.yr
        M0 = 10**(-7)*u.Msun/u.yr
        t = 3*u.Myr
        tvis = self.calculate_tvis(R1)
        return Mt - M0*(t/tvis + 1)**power
        # return 10**(np.log10(0.1)/power) - (3*u.Myr/self.calculate_tvis(R1)).decompose() - 1
    
    
    def calculate_R1(self):
        import scipy.optimize
        R1 = scipy.optimize.fsolve(self.f_root, 1)
        return R1[0]*u.AU


    def set_R1(self, R1):
        # I create this function because whenever R1 is changed, tvis should be changed.
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)    
    
    
    def get_R1(self):
        return self.__R1   
    
    
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)


    def vr_solid(self, r, t):
        return (self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St_).to(u.m/u.s)
        
    
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

        
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               

    def delta_v(self, r, t):
        # SubKeplerian Velocity
        
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
    
    
    def chi(self, r, t):
        T = 1 + (t-self.t0)/self.tvis
        return self.chi0 + (2-self.gamma)*(r/self.get_R1())**(2-self.gamma)/T







#------------------- Z(r, t) numerically, non-constant chi and St------------------


import scipy.interpolate


class ProtoplanetaryDisc4():
    
    gasDisipation = True # The gas in the disc decreases with time
    exponential = True # The profile of the gas density ends with an exponential
    St_cons = False # If true, Stokes number constant (and equal to alpha)
    
    def __init__(self, t0=0.2*u.Myr, alpha=0.01, St=0.03, delta=0.0001, Z0=0.01, R1=100*u.AU,
                 Mstar=1*u.solMass, Mdot=10**(-7)*u.solMass/u.yr, cs1=650*u.m/u.s, zeta=3./7, vf=1*u.m/u.s, lifetime=3*u.Myr):
        self.t0 = t0
        self.lifetime = lifetime
        self.Mstar = Mstar
        self.M_dot_0 = Mdot #self.calculate_M_dot_0()
        self.Z0 = Z0
        
        self.constant_flux = True # to avoid problems with planetesimal class
        self.alpha = alpha
        self.St_ = St
        self.delta = delta
        self.vf = vf
        self.cs1 = cs1
        self.zeta = zeta
        self.gamma = 3/2 - self.zeta
        self.kmig = 2*(1.36 + 0.62*self.gamma + 0.43*self.zeta) 
        self.chi0 = self.gamma + self.zeta/2 + 3/2
        self.xi = Z0*(1 + (2/3)*(St/alpha)*self.chi0)/(1 + St**2)  # to avoid problems
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)
        self.Z = self.Z_compute()
        
        
    def calculate_tvis(self, R1):
        return (R1**2/self.visc(R1)/3/(2-self.gamma)**2).to(u.Myr)
    
    
    def St(self, r):
        if self.St_cons:
            return self.St_
        else:
            return (((self.vf/self.cs(r))**2)/3/self.delta).decompose()


    def calculate_M_dot_0(self):
        M_dot = 10**-7
        return M_dot*u.solMass/u.yr
    
    
    def choose_model(self, gasDis=True, exponent=True, Stcons=False, Mdot=False, R1calculate=False):
        self.gasDisipation = gasDis
        self.exponential = exponent
        self.St_cons = Stcons
        if Mdot:
            self.M_dot_0 = 10**(-7)*u.solMass/u.yr
        if R1calculate:
            self.__R1 = self.calculate_R1()
            self.tvis = self.calculate_tvis(self.__R1)


    def initial_gas_function(self, r):
        return (self.M_dot_0/3/np.pi/self.visc(r)).to(u.g/u.cm**2)
    
        
    def sigma_g(self, r, t, tunit):
        # We assume that the there is only one possible solution for the density 
        result = self.initial_gas_function(r)
        if self.gasDisipation:
            T = (1 + (t*tunit-self.t0)/self.tvis)
            power = -(5/2-self.gamma)/(2-self.gamma)
            result *= T**power
        else:
            T = 1    
        if self.exponential:
            result *= np.exp(-((r/self.__R1)**(2-self.gamma))/T)     
        return result
    
    
    def H(self, r):
        return (self.cs(r)/self.kepler_angular(r)).to(u.AU)


    def cs(self, r):
        return (self.cs1*(r/u.AU)**(-self.zeta/2)).to(u.m/u.s)
    
    
    def visc(self, r):
        return self.alpha*self.cs(r)*self.H(r)

    
    def kepler_angular(self, r):
        return (G*self.Mstar/r**3)**(1/2)
    
    
    def sigma_p(self, r, t, tunit):
        return self.sigma_g(r, t, tunit)*self.Z(r, t*tunit)


    def Z_compute(self):
        rmin, rmax = 0.1*u.AU, 3000*u.AU
        tmin, tmax = self.t0, self.lifetime
        tau = 0.00001*u.Myr
        h = 1*u.AU
        ni, nj = int((tmax-tmin)/tau), int((rmax-rmin)/h)
        r_arr = np.linspace(rmin.value, rmax.value, nj)*u.AU
        t_arr = np.linspace(tmin.value, tmax.value, ni)*u.Myr
        Zij = np.zeros((ni, nj))
        Zij[:,-1] = 0

        Zij[0,:] = self.Z0

        r_nu = r_arr/self.get_R1()
        tau_T = tau/self.tvis
        h_nu = h/self.get_R1()

        for i in range(1, ni):
            Ti = (1 + (t_arr[i-1]-self.t0)/self.tvis)
            bj =  2*self.St(r_arr)/3/self.alpha*self.chi(r_arr, t_arr[i])
            mult = tau_T/(2-self.gamma)**2/r_nu**(1-self.gamma)
            A = (1/2)*(1 + bj[:-1] - 2*(2-self.gamma)*r_nu[:-1]**(2-self.gamma)/Ti)*(Zij[i-1,1:] - Zij[i-1,:-1])/h_nu 
            B = Zij[i-1, :-1]*(1/2)*(r_nu[:-1]**(self.gamma-1)*(2-self.gamma)*bj[:-1]/Ti - (bj[1:]-bj[:-1])/h_nu)
            Zij[i,:-1] = mult[:-1]*(A - B) + Zij[i-1, :-1]
            Zij[i, -1] = mult[-1]*(-(1/2)*Zij[i-1, -1]*(r_nu[-1]**(self.gamma-1)*(2-self.gamma)*bj[-1]/Ti))+ Zij[i-1, -1] 
        return scipy.interpolate.interp2d(r_arr, t_arr, Zij, kind='cubic')
        
    
    def disc_mass(self, t, tunit):
        import scipy.integrate        
        def f(r, t, tunit):
            return (r*(self.sigma_g(r*u.AU, t, tunit)).to(u.g/u.cm**2)).value
        return (2*np.pi*scipy.integrate.quad(f, 0.0001, np.Inf, args=(t, tunit))[0]*u.g/u.cm**2*u.AU**2).to(u.solMass)

    
    def f_root(self, R1):
        R1 = R1*u.AU
        power = -(5/2 - self.gamma)/(2-self.gamma)
        Mt = 10**(-8)*u.Msun/u.yr
        M0 = 10**(-7)*u.Msun/u.yr
        t = 3*u.Myr
        tvis = self.calculate_tvis(R1)
        return Mt - M0*(t/tvis + 1)**power
    
    
    def calculate_R1(self):
        import scipy.optimize
        R1 = scipy.optimize.fsolve(self.f_root, 1)
        return R1[0]*u.AU


    def set_R1(self, R1):
        self.__R1 = R1
        self.tvis = self.calculate_tvis(R1)    
    
    
    def get_R1(self):
        return self.__R1   
    
    
    def vr_gas(self, r, t):
        return -self.Mg_dot(r, t.value, t.unit)/2/np.pi/r/self.sigma_g(r, t.value, t.unit)


    def vr_solid(self, r, t):
        return ((self.vr_gas(r, t)-2*self.delta_v(r, t)*self.St(r))/(1 + self.St(r)**2)).to(u.m/u.s)
        
    
    def Mg_dot(self, r, t, tunit):
        T = (1 + (t*tunit-self.t0)/self.tvis)
        power = -(5/2-self.gamma)/(2-self.gamma)
        return self.M_dot_0*T**power*np.exp(-(r/self.__R1)**(2-self.gamma)/T)*(1-2*(2-self.gamma)/T*(r/self.__R1)**(2-self.gamma))

        
    def Mp_dot(self, r, t, tunit):
        return self.Mg_dot(r, t, tunit)*self.Z(r, t*tunit)*self.vr_solid(r, t*tunit)/self.vr_gas(r, t*tunit)
               

    def delta_v(self, r, t):
        return 0.5*(self.H(r)/r)*self.chi(r, t)*self.cs(r)
    
    
    def chi(self, r, t):
        T = 1 + (t-self.t0)/self.tvis
        return self.chi0 + (2-self.gamma)*(r/self.get_R1())**(2-self.gamma)/T



