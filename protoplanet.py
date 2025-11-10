import astropy.units as u
import numpy as np
from astropy.constants import G
from scipy.integrate import solve_ivp


class Protoplanet():
    gasAccretion = False # True if gas accretion activated
    tau_thr = False      # True if pebble decay pathway for gas accretion activated

    def __init__(self, disc, r0, M0=0.01*u.Mearth, kappa=0.005*u.m**2/u.kg, tau=100*u.Myr):
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
        kappa : float Length**2/mass units
            Opacity for gas accretion
        tau : float time
            Threshold timescale to start gas accretion by pebble decay pathway
        """
        self.disc = disc    # disc at which protoplanet forms
        self.r0 = r0        # initial position of embryo
        self.M0 = M0        # initial mass of embryo
        self.kappa = kappa  # standard value of opacity from Johansen 2019
        self.tau = tau      # threshold for the doubling time so that gas accretion starts
    
    
    def choose_model(self,threshold_gas=False, gasAccretion=False):
        """
        Set threshold_gas=True to activate gas accretion by pebble decay pathway
        Set gasAccretion=True to activate gas accretion

        """
        self.tau_thr = threshold_gas
        self.gasAccretion = gasAccretion

    # Hill radius
    def hill_radius(self, r, M):
        return ((M/3/self.disc.Mstar)**(1/3)*r).to(u.AU)
    
    # pebble scale height
    def Hp(self, r, t):
        return self.disc.H(r)*np.sqrt(self.disc.delta/(self.disc.delta + self.disc.St(r, t)))

    # transition mass between Bondi and Hill regime
    def Mt(self, r, t, tunit):
        return (25/144)*self.disc.delta_v(r, t*tunit)**3/G/self.disc.kepler_angular(r)/self.disc.St(r, t*tunit)
    
    # accretion radius in Hill regime
    def Racc_Hill(self, r, M, t, tunit):
        return (self.disc.St(r, t*tunit)/0.1)**(1/3)*self.hill_radius(r, M)

     # accretion radius in Bondi regime
    def Racc_Bondi(self, r, M, t, tunit):
        return (4*self.disc.St(r, t*tunit)/self.disc.kepler_angular(r)*G*M/self.disc.delta_v(r, t*tunit))**(1/2)
    
    # type I migration rate
    def typeI_migration(self, r, M, t, tunit):
        rdotI = -self.disc.kmig*(M/self.disc.Mstar)*(self.disc.sigma_g(r, t, tunit)*r**2/self.disc.Mstar)
        rdotI *= (self.disc.H(r)/r)**(-2)*(self.disc.kepler_angular(r)*r)
        return rdotI
        
    # migration and pebble accretion rate onto protoplanet
    def deriv_solid(self, t, y, tunit, runit, Munit):
        """
        ODEs for r (distance from protoplanet to star) and M (mass of protoplanet).
        As we use already implemented functions to solve the system  differential
        equations (solve_ivp() preferably), t and y are adimensionless. 
    
        Parameters
        ----------
        t : scalar
        y : ndarray (2,)
            r, M. Distance from protoplanet to star and the mass of the protoplanet.
        tunit : Astropy unit, time dim
        runit : Astropy unit, lenght dim
        Munit : Astropy unit, mass dim
    
        Returns
        -------
        rdot : float  (can also be array), adim
            Derivative of the distance.
        Mdot : float (can also be array), adim
            Derivative of the mass.
    
        """
        r, M = y[0]*runit, y[1]*Munit


        if isinstance(M.value, float):
            if M>self.Miso(r): return 0., 0. # check if protoplanet larger than Miso
            if M > self.Mt(r, t, tunit): # check if it is Bondi/Hill regime
                Racc = self.Racc_Hill(r, M, t, tunit)
            else:
                Racc = self.Racc_Bondi(r, M, t, tunit)
            d_v = self.disc.delta_v(r,t*tunit) + self.disc.kepler_angular(r)*Racc

            # 2D pebble accretion
            Mdot = 2*Racc*self.disc.sigma_p(r, t, tunit)*d_v
            
            # check if 3D accretion. If so, add correction
            RaccH_ratio = Racc/self.Hp(r, t*tunit)
            if RaccH_ratio < np.sqrt(8/np.pi):
                Mdot *= RaccH_ratio*np.sqrt(np.pi/8) # 3D accretion
                
            # check if accretion rate smaller than pebble flux.
            # condition should not be true
            if np.abs(Mdot.to(u.Mearth/u.yr)) > np.abs((self.disc.Mp_dot(r, t, tunit)).to(u.Mearth/u.yr)):
                print(f'CAUTION: Growth rate higher than pebble flux! {M:.2f}, {r:.2f}, {t:.2f}')
                Mdot = self.disc.Mp_dot(r, t, tunit)
        else:
            Racc = np.where(M > self.Mt(r, t, tunit),
                            self.Racc_Hill(r, M, t, tunit),
                            self.Racc_Bondi(r, M, t, tunit))
            d_v = self.disc.delta_v(r,t*tunit) + self.disc.kepler_angular(r)*Racc

            # 2D pebble accretion
            Mdot = 2*Racc*self.disc.sigma_p(r, t, tunit)*d_v
            
            # check if 3D accretion. If so, add correction
            RaccH_ratio = Racc/self.Hp(r, t*tunit)
            Mdot *= np.where(RaccH_ratio < np.sqrt(8/np.pi), 
                             RaccH_ratio*np.sqrt(np.pi/8), 1)
            
            # this should not occur
            Mdot = np.where(np.abs(Mdot.to(u.Mearth/u.yr)) > np.abs((self.disc.Mp_dot(r, t, tunit)).to(u.Mearth/u.yr)),
                     self.disc.Mp_dot(r, t, tunit), Mdot)
        
        # migration rate
        rdot = self.typeI_migration(r, M, t, tunit)
        # effect of gap oppening
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
        tunit : Astropy unit, time dim
        runit : Astropy unit, lenght dim
        Munit : Astropy unit, mass dim
    
        Returns
        -------
        float
            if <0 the simulation stops.
    
        """
        r = y[0]*runit
        Miso_ = self.Miso(r)
        return Miso_.to(Munit).value - y[1]
    M_equal_Miso.terminal = True
    
    # pebble isolation mass
    def Miso(self, r):
        Miso = (25*u.Mearth*(self.disc.H(r)/r/0.05)**3).to(u.Mearth)
        Miso *= (0.34*(np.log10(0.001)/np.log10(self.disc.delta))**4 + 0.66)*(1-(-self.disc.chi0+2.5)/6)
        return Miso.to(u.Mearth)
    
    
    def doublingMass(self, t, y, tunit, runit, Munit):
        M = y[1]*Munit
        Mdot = self.deriv_solid(t, y, tunit, runit, Munit)[1]*Munit/tunit
        
        # check if doubling mass timescale is longer than threshold timescale
        if (M/Mdot).to(u.Myr) > self.tau:
            return 0
        else:
            return 1
    doublingMass.terminal = True
    
    # migration and gas accretion rate onto protoplanet
    def deriv_gas(self, t, y, tunit, runit, Munit):
        r, M = y[0]*runit, y[1]*Munit

        # gas accretion rate towards star
        Mdot_g = np.abs(self.disc.Mg_dot(r, t, tunit)).copy()
     
        # rate at which gas enters the Hill sphere
        Mdot_disc = 1.5*10**(-3)*u.Mearth/u.yr*(self.disc.H(r)/r/0.05)**(-4)*(M/10/u.Mearth)**(4/3)
        Mdot_disc *= (self.disc.alpha/0.01)**(-1)*(Mdot_g/(10**(-8)*u.Msun/u.yr))/(1+ (M/2.3/self.Miso(r))**2)
        
        # rate of contraction of the gaseous envelope
        Mdot_kh = 10**(-5)*u.Mearth/u.yr*(M/10/u.Mearth)**4*(self.kappa/(0.1*u.m**2/u.kg))**(-1)

        Mdot = min(Mdot_kh.to(u.Mearth/u.yr), Mdot_disc.to(u.Mearth/u.yr), 0.8*Mdot_g.to(u.Mearth/u.yr))
        
        # migration rate
        rdot = self.typeI_migration(r, M, t, tunit)
        rdot /= (1+ (M/2.3/self.Miso(r))**2) # gap opening

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

        if self.tau_thr:
            soln = solve_ivp(self.deriv_solid, (t0.value, tf.value), y0, max_step=max_step,
                             args=(t0.unit, self.r0.unit, self.M0.unit),
                             events=[self.M_equal_Miso, self.doublingMass], method=method)
        else:
            soln = solve_ivp(self.deriv_solid, (t0.value, tf.value), y0, max_step=max_step,
                             args=(t0.unit, self.r0.unit, self.M0.unit),
                             events=self.M_equal_Miso, method=method)

        t, r, M = soln.t, soln.y[0], soln.y[1]
        
        if soln.status == 1 and self.gasAccretion == True:
            print(f'Gas accretion starting t is {t[-1]}')
            print(f'Gas accretion starting r is {r[-1]}')
            print(f'Gas accretion starting M is {M[-1]}')
            y0 = r[-1], M[-1]
            soln = solve_ivp(self.deriv_gas, (t[-1], tf.value), y0, max_step=max_step, args=(t0.unit, self.r0.unit, self.M0.unit))
            tg, rg, Mg = soln.t, soln.y[0], soln.y[1]

            return t*tf.unit, r*self.r0.unit, M*self.M0.unit, tg*tf.unit, rg*self.r0.unit, Mg*self.M0.unit
        else:
            return t*tf.unit, r*self.r0.unit, M*self.M0.unit, None, None, None
    


    
