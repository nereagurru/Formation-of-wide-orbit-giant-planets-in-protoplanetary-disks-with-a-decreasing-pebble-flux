#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:37:20 2025

@author: nerea
"""

from disc import ProtoplanetaryDisc_constZ
from disc import ProtoplanetaryDisc_Zt
from disc import ProtoplanetaryDisc_Zrt

import numpy as np
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt


t_arr = np.array([0.2, 0.6, 1, 2, 3])*u.Myr
r_arr  = np.arange(1, 1000, 0.1)*u.AU



# data from Appelgren+2023 without disc formation
data = np.loadtxt('dust_drift_rate_no_form.dat',  dtype=float,
                  delimiter='\t')


# from 2.7 AU to 4700 AU
r_arr_j = data[:, 0]

h = 0.001
t_arr_j = np.arange(0, 3+h, h)*u.Myr

# 77000Myr until the disk starts forming
Mp_j = data[:, :]
Mp_plot  = np.empty((r_arr_j.shape[0], t_arr.shape[0]))*u.Mearth/u.yr
i = 0
for j, tj in enumerate(t_arr_j):
    if np.abs((tj - t_arr[i]).value) <0.001:
        Mp_plot[:, i] = Mp_j[:, int(tj.value*1000)]*10**-6*u.Mearth/u.yr
        i +=1 
        if i == t_arr.shape[0]:
            break

disc0 = ProtoplanetaryDisc_constZ(Mdot=6*10**(-8)*u.solMass/u.yr, St=0.03, Z0=0.008)
disc1 = ProtoplanetaryDisc_Zt(Mdot=6*10**(-8)*u.solMass/u.yr, St=0.03, Z0=0.008)
disc2 = ProtoplanetaryDisc_Zrt(Mdot=6*10**(-8)*u.solMass/u.yr, St=0.03, Z0=0.008)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40, 15), sharex=True, sharey=True)
font = 22
init_plot(ax[0], r'$\textnormal{Constant } Z$', r'$\textnormal{r [AU]}$', 
          r'$\mathrm{\dot{\mathcal{M}}_{\rm{p}}\,[M_{\oplus}\,yr^{-1}]}$', font = font)
init_plot(ax[1], r'$\textnormal{Constant } \rm{St}_{\chi}$', r'$\textnormal{r [AU]}$', 
          None, font = font)

init_plot(ax[2], r'$\textnormal{Constant } \rm{St}$', r'$\textnormal{r [AU]}$', 
          None, font = font)
dim = t_arr.shape[0]
colors = ['#8BBCCC', '#3E6D9C', '#5C2E7E','darkred', '#000000']

for i, t in enumerate(t_arr):

    c = colors[i]# cmap(i/(dim-1))
    if i >= 2:
        num = 0
        alpha=0.8
    else:
        num = 1
        alpha=1

    
    for j, axx, dotMp in zip([0, 1, 2], ax.flatten(),
                          [(disc0.Mp_dot(r_arr, t.value, t.unit)).to(u.Mearth/u.yr),
                           (disc1.Mp_dot(r_arr, t.value, t.unit)).to(u.Mearth/u.yr),
                           (disc2.Mp_dot(r_arr, t.value, t.unit)).to(u.Mearth/u.yr)]):

        label = rf'${t.value:.{num}f}\,\mathrm{{Myr}}$' if j == 0 else None
        axx.plot(r_arr,dotMp,c=c, label=label,
                 linewidth=3.5, alpha=alpha, zorder=dim-i)
        axx.plot(r_arr_j, Mp_plot[:, i],
                   c=c, linewidth=2.5,
                   linestyle='--', alpha=alpha*0.5)

ax[1].plot([], [], c='gray', linewidth=3.5, label=r'$\textnormal{Analytical}$')
ax[1].plot([], [], c='gray', linestyle='--', linewidth=2.5, alpha=0.5, label=r'$\textnormal{Appelgren+ 2023}$')
ax[1].legend(loc='lower left', fontsize=font-2)
def fmt(x, pos):
    return rf'${x:.1f}$'

ax[0].set_xscale('log')

ax[0].set_yscale('log')

ax[0].set_xlim(1, 300)

ax[0].set_ylim(10**-8, 3*10**-3)
ax[0].set_yticks([10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3])
ax[0].legend(loc='lower left', ncol=2, fontsize=font-2)

plt.subplots_adjust(wspace=0.05, hspace=0.01, left=0.1, right=0.95, top=0.9, bottom=0.37)

fig.tight_layout()
fig.savefig('pebbleFluxComparison.pdf', dpi=500, format='pdf', bbox_inches='tight')
