
from disc import ProtoplanetaryDisc_constZ
from protoplanet import Protoplanet
from astropy.constants import G
import numpy as np
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib import ticker


St_max = 0.03
disc = ProtoplanetaryDisc_constZ(St=St_max)
protoplanet = Protoplanet(disc, r0=None)
r = np.arange(10, 150, 0.5)*u.AU
t0 = 0.2*u.Myr

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
init_plot(ax, None , r'$\textnormal{r [AU]}$', r'$\mathrm{M\;[M_{\oplus}]}$', font=30)

M = np.arange(0.01, 10, 0.001)*u.Mearth #Mt.min().value, Mt.max().value, 0.001)*u.Mearth

r_arr, M_arr = np.meshgrid(r, M)
Mt = protoplanet.Mt(r, t0.value, t0.unit)

ax.plot(r, Mt.to(u.Mearth), 'white',linestyle='-', linewidth=3)

ax.set_xscale('log')
ax.set_yscale('log')

Mstar = disc.Mstar
M_h = (0.3*Mstar/St_max)*(disc.H(r)/r*np.sqrt(8/np.pi*disc.delta/(disc.delta+St_max)))**3
M_b = disc.delta_v(r, t0.value)/G*disc.kepler_angular(r)/St_max*(2/np.pi)*disc.delta/(disc.delta + St_max)*disc.H(r)**2

bool_h = M_h >  Mt
M_2d3d = np.hstack((M_h.to(u.Mearth).value[bool_h], M_b.to(u.Mearth).value[np.logical_not(bool_h)]))
ax.plot(r.value[bool_h], M_h.to(u.Mearth)[bool_h], 'gold', linestyle=':', linewidth=2)
ax.plot(r.value[np.logical_not(bool_h)], M_b.to(u.Mearth)[np.logical_not(bool_h)], 'gold', linestyle=':', linewidth=2)


Mdot = (protoplanet.deriv_solid(t0.value, [r_arr.value, M_arr.value],
                                u.Myr, u.AU, u.Mearth)[1])*u.Mearth/u.Myr
levels = np.array([10**-7, 5*10**-7, 10**-6, 5*10**-6, 10**-5, 5*10**-5, 10**-4, 5*10**-4, 10**-3])


def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${}\times10^{{{}}}$'.format(a, b)



plot_2d = ax.contourf(r_arr, M_arr, Mdot.to(u.Mearth/u.yr).value, levels=levels,
                      norm=LogNorm(vmin=Mdot.min().to(u.Mearth/u.yr).value, vmax=0.001),
                      cmap=sns.color_palette("mako_r", as_cmap=True))

#ax.fill_between(r.value, 0, Mt.to(u.Mearth).value, color='k', alpha=0.5)

print(Mdot.min().to(u.Mearth/u.yr).value)
print(Mdot.max().to(u.Mearth/u.yr).value)



cbar = fig.colorbar(plot_2d, ax=ax, format=ticker.FuncFormatter(fmt), extend='both')
cbar.ax.set_title(r'$\mathrm{\dot{M}\,[M_{\oplus}\,yr^{-1}}] $', fontsize=25, pad=20)

cbar.ax.yaxis.set_ticks_position('both')
cbar.ax.tick_params(labelsize=25, color='k')


# 2d and 3d

# First we find point r0 where Mt (so Racc euqal in both regimes)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5, edgecolor='white', linestyle='-')
props1 = dict(boxstyle='round', facecolor='khaki', alpha=0.5, edgecolor='gold', linestyle=':', linewidth=2)
ax.text(x=25, y=0.225, s=r'$\textnormal{Hill}$', color='white', fontsize=30, bbox=props, rotation=30)
ax.text(x=28, y=0.1, s=r'$\textnormal{Bondi}$', color='white', fontsize=30, bbox=props, rotation=30)

ax.text(x=13.7, y=0.17, s=r'$\textnormal{3D}$', color='white', fontsize=20, bbox=props1, rotation=15)
ax.text(x=13, y=0.33, s=r'$\textnormal{2D}$', color='white', fontsize=20, bbox=props1, rotation=15)


ax.set_xlim(r.min().value, r.max().value) #10.0, 149.5)
ax.set_ylim(0.01, 10) #Mt.min().value, Mt.max().value) # 0.029131113209302197, 3.0062680200646814)


fig.savefig('Hill_Bondi.pdf', dpi=500, format='pdf',bbox_inches='tight')

