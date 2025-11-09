# script to plot Fig. 1
import numpy as np
from disc import ProtoplanetaryDisc_constZ as ProtoplanetaryDisc
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt
from matplotlib import ticker


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 30), sharex=True)
init_plot(ax[0], None, None , r'$\mathrm{\Sigma_{\rm{g}}\,[g\,cm^{-2}]}$', font = 30)
init_plot(ax[1], None, r'$\textnormal{r [AU]}$', r'$\mathrm{\dot{\mathcal{M}}_{\rm{g}}\,[M_{\odot}\,yr^{-1}]}$', font = 30)
R1 = 100*u.AU
disc = ProtoplanetaryDisc(alpha=0.01, R1=R1, t0=0.2*u.Myr)


r_arr  = np.arange(5, 1000, 0.1)*u.AU
t_arr = np.array([0.2, 0.6, 1, 2, 3])*u.Myr

colors = plt.cm.inferno(np.linspace(0, 1, t_arr.shape[0]))
norm = plt.Normalize(vmin=disc.t0.value, vmax=3) 
dim = t_arr.shape[0]
colors = ['#8BBCCC', '#3E6D9C', '#5C2E7E','darkred', '#000000']

# location at which Mdot=0
Rt = R1*((1 + (t_arr-disc.t0)/disc.tvis)/2/(2-disc.gamma))**(1/(2-disc.gamma))

ax[1].axhline(0, linestyle='-.', c='lightgray', zorder=0)

for i, t in enumerate(t_arr):

    c = colors[i]
    if i >= 2:
        num = 0
        alpha=0.8
    else:
        num = 1
        alpha=1
    ax[0].plot(r_arr, (disc.sigma_g(r_arr, t.value, t.unit)).to(u.g/u.cm**2),
               c=c, linewidth=3, alpha=alpha, zorder=t_arr.shape[0]-i)
    ax[1].plot(r_arr, (disc.Mg_dot(r_arr, t.value, t.unit)).to(u.Msun/u.yr),
               c=c, label=rf'${t.value:.{num}f}\,\mathrm{{Myr}}$', linewidth=3,
               alpha=alpha, zorder=t_arr.shape[0]-i)
    ax[0].scatter(Rt[i], (disc.sigma_g(Rt[i], t.value, t.unit)).to(u.g/u.cm**2),
                  c=c, zorder=t_arr.shape[0]-i)
    ax[1].scatter(Rt[i], (disc.Mg_dot(Rt[i], t.value, t.unit)).to(u.Msun/u.yr),
                  c=c, zorder=t_arr.shape[0]-i)





def fmt(x, pos):
    return rf'${x:.1f}$'

ax[0].set_xscale('log')
ax[0].set_yscale('log')


ax[0].set_xlim(5, 300)
ax[0].set_ylim(0.1, 10**3)
ax[1].set_ylim(-5*10**(-8), 10*10**(-8))
def fmt2(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if b==0:
        return rf'${x:.0f}$'
    else:
        return r'${}\cdot 10^{{{}}}$'.format(a, b)
ax[0].secondary_xaxis('top')
ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(fmt2))


formatter = ticker.ScalarFormatter()
formatter.set_powerlimits((-1,1))
ax[1].yaxis.set_major_formatter(formatter)
ax[1].legend(loc='upper right', ncol=2, fontsize=21)



plt.subplots_adjust(wspace=0.0, hspace=0.2, left=0.15, right=0.7, top=0.9, bottom=0.12)
fig.savefig('density&Mdot.pdf', dpi=500, format='pdf', bbox_inches='tight')
