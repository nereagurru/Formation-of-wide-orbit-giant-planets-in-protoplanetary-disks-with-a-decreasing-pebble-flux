# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:10:55 2022

@author: nerea
"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.integrate import solve_ivp
from Plotting import init_plot


class Planetesimal():
    
    Mmax = None
    twoD = False
    allow_3D = True
    migration_gap = True
    gasAccretion = False
    BondiRegime = True
    tau_thr = False

    def __init__(self, disc, r0, fplt=400,  kappa=0.005*u.m**2/u.kg, tau=100*u.Myr):
        """
        Initialise the planetesimal.

        Parameters
        ----------
        disc : ProtoplanetaryDisc
            Protoplanet where the planetesimal is located.
        r0 : float   Lenght units
            Initial position of the planetesimal
        M0 : float  Mass units or None
            Initial mass of the planetesimal. If None, calculated by M0_calculate().
        fplt : float
            Ratio between the maximum mass and the characteristic mass fromed by Streaming instability
        """
        self.r0 = r0
        self.disc = disc
        self.fplt = fplt
        self.M0 = self.M0_calculate()
        self.kappa = kappa # standar value of opacity from Johansen 2019
        self.tau = tau #threshold for the doubling time so that gas accretion starts
       

    def gamma(self):
        """
            Dimensionless gravity-parameter: measures the relative strength
            between the self-gravity and tidal shear
        """
        gamma_grab = 4*np.pi*G*self.disc.sigma_g(self.r0, self.disc.t0.value, self.disc.t0.unit)/self.disc.H(self.r0)/np.sqrt(2*np.pi)/(self.disc.kepler_angular(self.r0))**2
        return gamma_grab.decompose()
        
    
    def M0_calculate(self):
        """ eq. 13/14 from Liu2020"""
        M_0 = 2*10**(-3)*(self.disc.Mstar/(0.1*u.solMass))*(self.disc.H(self.r0)/self.r0/0.05)**3*(self.gamma()*np.pi)**1.5*u.Mearth
        M_0 *= (self.disc.Z0/0.02)**0.5*(self.fplt/400)
        return M_0.to(u.Mearth)
    
    
    def choose_model(self, allow_3D=True, M0cons=False, migration_gap=False, gasAccretion=False, Bondi=False, threshold_gas=False):
        self.allow_3D = allow_3D
        self.twoD = np.logical_not(allow_3D)
        if M0cons:
            # u.Mearth = 10**24*u.kg  # In Johansen 2019 they use only one significant number
            self.M0 = 0.01*u.Mearth
        self.migration_gap = migration_gap
        self.gasAccretion = gasAccretion
        self.BondiRegime = Bondi
        self.tau_thr = threshold_gas


    def hill_radius(self, r, M):
        return ((M/3/self.disc.Mstar)**(1/3)*r).to(u.AU)
    
    
    def calculate_Mmax(self):
        """
        From "How planetary growth outperforms migration":
        Calculate Mmax using eq. 23.
    
        Parameters
        ----------
    
        Returns
        -------
        Mmax : float, mass dim
            Maximum mass that the protoplanet can have due to pebble accretion
            and migration of type I.
    
        """    

        Mmax = 11.7*u.Mearth*(self.disc.St_/0.01)**0.5
        Mmax /= (((2/3)*(self.disc.St_/self.disc.alpha)*self.disc.chi0 + 1)/2.9)**(3/4)
        Mmax *= (self.disc.xi/0.01)**(3/4)*(self.disc.Mstar/u.solMass)**(1/4)
        Mmax *= (self.disc.kmig/4.42)**(-3/4)*(self.disc.cs1/(650*u.m/u.s))**(3/2)
        Mmax *= (4/7/(1-self.disc.zeta))*(self.r0/(25*u.AU))**((3/4)*(1-self.disc.zeta))
        self.Mmax = Mmax.to(u.Mearth)
        return Mmax

    
    def rM_analytical(self, npoints=100):
        """
        From "How planetary growth outperforms migration":
        Calculate distance of the protoplanet from the star using eq. 24.
        Growth tracks are calculated only considering pebble accretion and migration
        of type I.
    
        Parameters
        ----------
        npoints : integer
            Number of points in the analytical data. 100 points by default.
    
        Returns
        -------
        float or numpy, lenght dim
            Position of the protoplanet for a given mass and initial conditions,
            computed analytically.
        float or numpy, lenght mass
            Mass of the protoplanet
    
        """
        if self.Mmax == None:
            self.calculate_Mmax()
        M = np.linspace(self.M0, self.Mmax, npoints)
        r = self.r0*(1 - (M**(4/3)-self.M0**(4/3))/(self.Mmax**(4/3)-self.M0**(4/3)))**(1/(1-self.disc.zeta))
        return r, M
    
    
    def K(self, r, M):
        return (M/self.disc.Mstar)**2*(self.disc.H(r)/r)**-5/self.disc.delta
  
    
    def deriv_solid(self, t, y, tunit, runit, Munit):
        """
        ODEs for r (distance from protoplanet to star) and M (mass of protoplanet).
        As we use already implemented functions to solve the system  differential
        equations (solve_ivp() preferably), t and y are adimensionless. 
    
        Parameters
        ----------
        t : scalar
        y : ndarray (2,)
            Distance from protoplanet to star and the mass of the protoplanet.
        den_gas : float, mass/lenght^2 dim 
            Surface density.
        St : float, dimless
            Stokes number to characterise the behaviour of solid within the gas.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        xi : float, dimless
            The ratio of fluxes, defined in eq. 15.
        Mstar : float, mass dim
            The mass pf the embryon star
        kmig : float, dimless
            Constant prefactor that depends on the gradients of surface density
            and temperature, defined in eq. 4.
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        tunit : Astropy unit, time dim
        runit : Astropy unit, lenght dim
        Munit : Astropy unit, mass dim
    
        Returns
        -------
        rdot : float, adim
            Derivative of the distance.
        Mdot : float, adim
            Derivative of the mass.
    
        """
        r, M = y[0]*runit, y[1]*Munit

        # To check if it is still Bondi
        if self.BondiRegime:
            Mt = (25/144)*self.disc.delta_v(r, t*tunit)**3/G/self.disc.kepler_angular(r)/self.disc.St(r, t*tunit)
            if M > Mt:
                self.BondiRegime = False

        if self.BondiRegime:
            Racc = (4*self.disc.St(r, t*tunit)/self.disc.kepler_angular(r)*G*M/self.disc.delta_v(r, t*tunit))**(1/2)
        else: 
            Racc = (self.disc.St(r, t*tunit)/0.1)**(1/3)*self.hill_radius(r, M)
        d_v = self.disc.delta_v(r,t*tunit) + self.disc.kepler_angular(r)*Racc

        Mdot = 2*Racc*self.disc.sigma_p(r, t, tunit)*d_v

        
        # Once that the accretion is 2D it cannot be 3D
        if self.twoD == False:
            #Mdot 3D

            RaccH_ratio = np.sqrt(8/np.pi)*Racc/self.Hp(r, t*tunit)
            # If Racc/Hp < 1, M3D. Once M2D, we don't need to check again.
            if RaccH_ratio < 1:
                Mdot *= RaccH_ratio*np.sqrt(np.pi/8)
            else:
                self.twoD = True
                
        if np.abs(Mdot.to(u.Mearth/u.yr)) > np.abs((self.disc.Mp_dot(r, t, tunit)).to(u.Mearth/u.yr)):
            print('true')
            Mdot = self.disc.Mp_dot(r, t, tunit)
        #eq. 3
        rdot = -self.disc.kmig*(M/self.disc.Mstar)*(self.disc.sigma_g(r, t, tunit)*r**2/self.disc.Mstar)
        rdot *= (self.disc.H(r)/r)**(-2)*(self.disc.kepler_angular(r)*r)
        if self.migration_gap:
            rdot /= (1+ (M/2.3/self.Miso(r))**2)
        return (rdot*tunit/runit).decompose(), (np.abs(Mdot)*tunit/Munit).decompose()
    

    def M_equal_Miso(self, t, y, tunit, runit, Munit):
        """
        Event to stop the simulation when M=Miso.
    
        Parameters
        ----------
        t : scalar
        y : ndarray (2,)
            Distance from protoplanet to star and the mass of the protoplanet.
        den_gas : float, mass/lenght^2 dim 
            Surface density.
        St : float, dimless
            Stokes number to characterise the behaviour of solid within the gas.
        alpha : float, dimless
            constant value of alpha (alpha-disk assumption).
        xi : float, dimless
            The ratio of fluxes, defined in eq. 15.
        Mstar : float, mass dim
            The mass pf the embryon star
        kmig : float, dimless
            Constant prefactor that depends on the gradients of surface density
            and temperature, defined in eq. 4.
        cs1 : float, velocity dim
            The sound speed at 1 AU.
        zeta : float, dimless
            The negative power-law index of the temperature (proportional to cs**2).
        tunit : Astropy unit, time dim
        runit : Astropy unit, lenght dim
        Munit : Astropy unit, mass dim
    
        Returns
        -------
        float
            if 0, the simulation stops.
    
        """
        r = y[0]*runit
        Miso_ = self.Miso(r)
        return round(Miso_.to(Munit).value - y[1], 4)
    
    
    def Miso(self, r):
        Miso = (25*u.Mearth*(self.disc.H(r)/r/0.05)**3).to(u.Mearth)
        Miso *= (0.34*(np.log10(0.001)/np.log10(self.disc.delta))**4 + 0.66)*(1-(-self.disc.chi0+2.5)/6)
        return Miso.to(u.Mearth)

    M_equal_Miso.terminal = True

    
    def flux_stop(self, t, y, tunit, runit, Munit):
        r = y[0]*runit
        if self.disc.last_pebble_pos(t*tunit) < r:
            return 0
        else:
            return 1
    flux_stop.terminal = True
    
    
    def doublingMass(self, t, y, tunit, runit, Munit):
        r, M = y[0]*runit, y[1]*Munit


        # To check if it is still Bondi
        if self.BondiRegime:
            Mt = (25/144)*self.disc.delta_v(r, t*tunit)**3/G/self.disc.kepler_angular(r)/self.disc.St(r, t*tunit)
            if M > Mt:
                self.BondiRegime = False

        if self.BondiRegime:
            Racc = (4*self.disc.St(r, t*tunit)/self.disc.kepler_angular(r)*G*M/self.disc.delta_v(r, t*tunit))**(1/2)
        else: 
            Racc = (self.disc.St(r, t*tunit)/0.1)**(1/3)*self.hill_radius(r, M)

        d_v = self.disc.delta_v(r,t*tunit) + self.disc.kepler_angular(r)*Racc

        Mdot = 2*Racc*self.disc.sigma_p(r, t, tunit)*d_v

        
        # Once that the accretion is 2D it cannot be 3D
        if self.twoD == False:
            #Mdot 3D

            RaccH_ratio = np.sqrt(8/np.pi)*Racc/self.Hp(r, t*tunit)
            # If Racc/Hp < 1, M3D. Once M2D, we don't need to check again.
            if RaccH_ratio < 1:
                Mdot *= RaccH_ratio*np.sqrt(np.pi/8)
            else:
                self.twoD = True
                
        if np.abs(Mdot.to(u.Mearth/u.yr)) > np.abs((self.disc.Mp_dot(r, t, tunit)).to(u.Mearth/u.yr)):
            print('true')
            Mdot = self.disc.Mp_dot(r, t, tunit)

        if (M/Mdot).to(u.Myr) > self.tau:
            return 0
        else:
            return 1
    doublingMass.terminal = True
    
    
    def deriv_gas(self, t, y, tunit, runit, Munit):
        r, M = y[0]*runit, y[1]*Munit

        Mdot_g = np.abs(self.disc.Mg_dot(r, t, tunit)).copy()
     
        Mdot_disc = 1.5*10**(-3)*u.Mearth/u.yr*(self.disc.H(r)/r/0.05)**(-4)*(M/10/u.Mearth)**(4/3)
        Mdot_disc *= (self.disc.alpha/0.01)**(-1)*(Mdot_g/(10**(-8)*u.Msun/u.yr))/(1+ (M/2.3/self.Miso(r))**2)
        #Mdot_disc = 0.29/3/np.pi*(self.disc.H(r)/r)**(-4)*(M/self.disc.Mstar)**(4/3)*(Mdot_g/self.disc.alpha)*(1/(1 + 0.04*self.K(r, M)))
        Mdot_kh = 10**(-5)*u.Mearth/u.yr*(M/10/u.Mearth)**4*(self.kappa/(0.1*u.m**2/u.kg))**(-1)

        Mdot = min(Mdot_kh.to(u.Mearth/u.yr), Mdot_disc.to(u.Mearth/u.yr), 0.8*Mdot_g.to(u.Mearth/u.yr))
        rdot = -self.disc.kmig*(M/self.disc.Mstar)*(self.disc.sigma_g(r, t, tunit)*r**2/self.disc.Mstar)
        rdot *= (self.disc.H(r)/r)**(-2)*(self.disc.kepler_angular(r)*r)
        
        if self.migration_gap:
            rdot /= (1+ (M/2.3/self.Miso(r))**2)
        return (rdot*tunit/runit).decompose(), (Mdot*tunit/Munit).decompose()
    
    
    def rM_numerical(self, t0, tf, max_step=0.001, method='RK45'):
        """
        Calculate numerically r (distance from protoplanet to star) and M
        (mass of protoplanet) over the time.
    
        Parameters
        ----------
        t0 : float, time dim
            Starting time
        tf : float, time dim
            Ending time
        r0 : float, lenght dim
            Initial position of the protoplanet
    
        Returns
        -------
        TYPE ndarray (n_points), time dim
            Time points.
        TYPE ndarray (n_points), lenght dim
            Position of the protoplanet over the time.
        TYPE ndarray (n_points), mass dim
            Mass of the protoplanet over the time.
    
        """
        y0 = self.r0.value, self.M0.value
        if self.allow_3D:
            self.twoD = False
        else:
            self.twoD = True
        
        if self.disc.constant_flux == True:
            if self.tau_thr:
                soln = solve_ivp(self.deriv_solid, (t0.value, tf.value), y0, max_step=max_step,
                                 args=(t0.unit, self.r0.unit, self.M0.unit),
                                 events=[self.M_equal_Miso, self.doublingMass], method=method)
            else:
                soln = solve_ivp(self.deriv_solid, (t0.value, tf.value), y0, max_step=max_step,
                                 args=(t0.unit, self.r0.unit, self.M0.unit),
                                 events=self.M_equal_Miso, method=method)
        else:
            soln = solve_ivp(self.deriv_solid, (t0.value, tf.value), y0, max_step=max_step,
                             args=(t0.unit, self.r0.unit, self.M0.unit),
                             events=[self.M_equal_Miso, self.flux_stop], method=method)
        t, r, M = soln.t, soln.y[0], soln.y[1]
        
        if soln.status == 1 and self.gasAccretion == True:
            print(f'The starting t is {t[-1]}')
            print(f'The starting r is {r[-1]}')
            print(f'The starting M is {M[-1]}')
            y0 = r[-1], M[-1]
            soln = solve_ivp(self.deriv_gas, (t[-1], tf.value), y0, max_step=max_step, args=(t0.unit, self.r0.unit, self.M0.unit))
            tg, rg, Mg = soln.t, soln.y[0], soln.y[1]
            return t*tf.unit, r*self.r0.unit, M*self.M0.unit, tg*tf.unit, rg*self.r0.unit, Mg*self.M0.unit
        else:
            return t*tf.unit, r*self.r0.unit, M*self.M0.unit, None, None, None
    
    def Hp(self, r, t):
        return self.disc.H(r)*np.sqrt(self.disc.delta/(self.disc.delta + self.disc.St(r, t)))

