from disc import ProtoplanetaryDisc_Zt
import numpy as np
from protoplanet import Protoplanet
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle


# --- Check convergence ---
def has_converged(r_values, idx, eps):
    """Return True if spacing around the peak is smaller than eps."""
    if len(r_values) < 3:  # handle 1 or 2 valid points
        return False
    if idx == 0 or idx == len(r_values) - 1:
        return False
    left_diff = abs(r_values[idx] - r_values[idx-1])
    right_diff = abs(r_values[idx+1] - r_values[idx])
    print(f'{right_diff} and {left_diff}')
    return max(left_diff, right_diff) < eps


# eps =1 AU; accuracy at which we find furthest core
def find_distant_core(disc, t0p, r0min, r0max, tf=3*u.Myr,
                      N=5, eps=1, kont_max=10, max_step=0.01):
    
    kont = 0
    r0 = np.linspace(r0min.value, r0max.value, N)
    while True:
        ts = np.zeros(r0.shape, dtype=object) # in Myr
        rs = np.zeros(r0.shape, dtype=object) # in AU
        Ms = np.zeros(r0.shape, dtype=object) # in Mearth
    
        # Run through each r0 value
        for i, r in enumerate(r0*u.AU):
            protoplanet = Protoplanet(disc, r)
            ts_, rs_, Ms_, _, _, _ , sol_= protoplanet.rM_numerical(t0p, tf, max_step=max_step,
                                                                    return_status=True)
            print(f'{sol_} {ts_[-1]} {r}')
            # did it reach the pebble isolation mass?
            if sol_ == 1:
                ts[i], rs[i], Ms[i] = ts_[-1].value, rs_[-1].value, Ms_[-1].value
            else:
                ts[i], rs[i], Ms[i] = None, None, None
    
        # Filter out None values
        print(Ms)
        valid_indices = [i for i, m in enumerate(Ms) if m is not None]
        if len(valid_indices)==0:
            print("No valid Ms values found.")
            return None, None, None, None

        
    
        imax_valid = np.argmax(rs[valid_indices]) # furthest code
        imax = valid_indices[imax_valid] # map back to original index
        r0_valid = r0[valid_indices]
        
        # Return results if convergence is achieved
        if has_converged(r0_valid, imax_valid, eps):
            print(f"Converged at r0 = {r0[imax]} after {kont} iterations")
            print("t =", ts[imax])
            print("M =", Ms[imax])
            print("r =", rs[imax])
            return r0[imax], ts[imax], Ms[imax], rs[imax]

        # If did not converge
        if imax_valid == 0 or imax_valid == len(r0_valid) - 1:
            kont += 1
            if kont > kont_max:
                print("Reached max iterations without convergence")
                return None, None, None, None
            print(f"Refining near boundary (attempt {kont})")

  
        # check converce of results
        if len(valid_indices) == 1:
            if imax == 0:
                r0 = np.linspace(r0[0]-(r0[1]-r0[0]), r0[1], N)
            elif imax == len(r0) - 1:
                r0 = np.linspace(r0[-2], r0[-1]+(r0[-1]-r0[-2]), N)
            else:
                r0 = np.linspace(r0[imax_valid-1], r0[imax_valid+1], N)
 
        else:
            if imax_valid == 0:
                r0 = np.linspace(r0_valid[0]-(r0_valid[1]-r0_valid[0]), r0_valid[1], N)
            elif imax_valid == len(r0_valid) - 1:
                r0 = np.linspace(r0_valid[-2], r0_valid[-1]+(r0_valid[-1]-r0_valid[-2]), N)
            else:
                r0 = np.linspace(r0_valid[imax_valid-1], r0_valid[imax_valid+1], N)
        
        print("Refined r0 =", r0)

        

# this class will save the parameters of different type of simulations    
class Sim():
    def __init__(self, label, t0p=0.2*u.Myr, Z0=0.01, St=0.03, delta=10**-4):
        self.label = label
        self.Z0 = Z0
        self.St = St
        self.delta = delta
    def print_sim(self):
        print(f'{self.label} simulation with Z0={self.Z0}, St={self.St}, delta={self.delta}')
        


class MyClass():
    def __init__(self, param):
        self.param = param

def save_object(obj):
    try:
        with open(f"furthestcore{t0p}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


#%% 


sim_dict = dict()
sim_label = [r'$\mathrm{lst}$', r'$\mathrm{fid}$', r'$\mathrm{hst}$', r'$\mathrm{lst.hz}$',
             r'$\mathrm{hz}$', r'$\mathrm{hst.hz}$', r'$\mathrm{lst.l}\alpha$',
             r'$\mathrm{l}\alpha$', r'$\mathrm{hst.l}\alpha$']
Z0_arr = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
St_arr = [0.01, 0.03, 0.06, 0.01, 0.03, 0.06, 0.01, 0.03, 0.06]
delta_arr = [10**-4, 10**-4, 10**-4, 10**-4, 10**-4, 10**-4, 10**-5, 10**-5, 10**-5]
t0p_arr = [0.2, 0.3, 0.5, 0.7]*u.Myr

y_pos = np.arange(len(sim_dict.keys()))[::-1]



R1_arr = [100, 300]*u.AU

for t0p in t0p_arr:
    for label, Z0, St, delta in zip(sim_label, Z0_arr, St_arr, delta_arr):
        sim_dict[label] = Sim(label=label, t0p=t0p,
                              Z0=Z0, St=St, delta=delta)

    r0_arr = np.zeros((len(sim_label),len(R1_arr)))*u.AU
    rs_arr = np.zeros((len(sim_label),len(R1_arr)))*u.AU
    Max_arr = np.zeros((len(sim_label),len(R1_arr)))*u.Mearth
    ts_arr = np.zeros((len(sim_label),len(R1_arr)))*u.Myr
    
    for i, sim in enumerate(sim_dict.keys()):
        sim_dict[sim].print_sim()
        for j, R1 in enumerate(R1_arr):
            r0min, r0max = 5*u.AU, R1
            disc = ProtoplanetaryDisc_Zt(St=sim_dict[sim].St, Z0=sim_dict[sim].Z0, delta=sim_dict[sim].delta, R1=R1)
            r0, ts, Ms, rs =  find_distant_core(disc, t0p, r0min, r0max,
                                                tf=2*u.Myr, N=5, eps=1, kont_max=10,
                                                max_step=0.1)
            r0_arr[i,j], ts_arr[i,j], Max_arr[i,j], rs_arr[i,j] = r0*u.AU, ts*u.Myr, Ms*u.Mearth, rs*u.AU
            print('---------------')
    obj = MyClass([sim_dict, r0_arr, ts_arr, Max_arr, rs_arr])
    save_object(obj)


#%% save data



#%%
two_R1 = True
plot_observations=True
plot_many = True
c= ['orange', 'orangered', '#C5AB6E', '#6081FF', 'gray', 'whitesmoke']

label = [r'$\mathrm{lst}$', r'$\mathrm{fid}$', r'$\mathrm{hst}$', r'$\mathrm{lst.hz}$',
             r'$\mathrm{hz}$', r'$\mathrm{hst.hz}$', r'$\mathrm{lst.l}\alpha$',
             r'$\mathrm{l}\alpha$', r'$\mathrm{hst.l}\alpha$']

if plot_many:
    fig, axx = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(120, 120))
    for i, ax in enumerate(axx.flatten()):
        ylabel = r'$\rm{r_{core}\,[AU]}$' if i == 0 or i == 2 else None
        xlabel = r'$\rm{Simulation}$' if i == 2 or i == 3 else None
        init_plot(ax, None,
                  xlabel,ylabel, font=19)

else:
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(60, 60))
    axx = np.array([ax])
    init_plot(axx[0], rf'$\rm{{t_{{0, p}}}}={t0p.value:.1f}\,\mathrm{{Myr}}$ ',
              r'$\rm{Simulation}$', r'$\rm{r_{core}\,[AU]}$', font=25)


for j, ax in enumerate(axx.flatten()):
    disc = ProtoplanetaryDisc_Zt(St=sim_dict[sim].St,
                                 Z0=sim_dict[sim].Z0,
                                 R1=R1, delta=sim_dict[sim].delta)

    obj = load_object(f"furthestcore{t0p_arr[j]}.pickle")

    sim_dict, r0_arr, ts_arr = obj.param[0], obj.param[1], obj.param[2]
    Max_arr, rs_arr = obj.param[3], obj.param[4]

    alpha = 0.9
    if two_R1:
        for i, sim in enumerate(sim_dict.keys()):
            ax.bar(label[i], rs_arr[i,0], color='#FFC26F', width=0.5, alpha=alpha, linewidth=2,
                   edgecolor='#C38154', zorder=0.2)
            ax.bar(label[i], rs_arr[i,1], color='#AAB7CB', width=0.5, alpha=alpha, linewidth=2,
                   edgecolor='#68758D', zorder=0.1)
            if j==3 and i == 0:
                ax.bar(label[i],0, color='#FFC26F', width=0.5, alpha=alpha, linewidth=2,
                        edgecolor='#C38154', label='$R_{1}=100\,\mathrm{AU}$')
                ax.bar(label[i],0, color='#AAB7CB', width=0.5, alpha=alpha, linewidth=2,
                        edgecolor='#68758D', zorder=0.1, label='$R_{1}=300\,\mathrm{AU}$')
    
                ax.legend(fontsize=18, loc='upper left')
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_ylim(0, 150)
            if i == 0:
                text = rf'$\mathrm{{t_{{0, p}}}}={t0p_arr[j].value:.1f}\,\mathrm{{Myr}}$ '
                ax.text(0.97, 0.93, text, ha='right', va='top', transform=ax.transAxes)
                
marnum = 5
ax.set_ylim(10, 260)
ax.set_yscale('log')

for i, ax in enumerate(axx.flatten()):
    if i == 0 or i == 1:
        secax = ax.secondary_xaxis('top')
        secax.set_xticks(ax.get_xticks())
        secax.tick_params(direction='in')
        secax.set_xticklabels(label)

if plot_observations:
    data = np.loadtxt('Exoplanets_Bae.rtf',  dtype=np.dtype('U'),
                      delimiter='$', skiprows=9)

    planets = np.empty((data.shape[0], 2))
    for j, ax in enumerate(axx.flatten()):
        for i in range(0, data.shape[0]):
            dat = data[i].split('-')
            planets[i, 0], planets[i, 1]  = float(dat[2]), float(dat[1])
            ax.axhline(planets[i,0], color='#C3EDC0', linestyle ='-', zorder=0, alpha=0.3)
        
        if j==1: ax.text(-0.2, 180, r'$\textnormal{ALMA planet candidates}$',color='#B3DAB1', fontsize=18)

plt.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.1,
                    top=0.9, left=0.1, right=0.95) 
