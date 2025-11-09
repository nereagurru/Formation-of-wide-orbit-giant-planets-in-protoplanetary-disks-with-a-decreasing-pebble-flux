#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:14:05 2025

@author: nerea
"""
from disc import ProtoplanetaryDisc_Zt
from disc import ProtoplanetaryDisc_Zrt
import numpy as np
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt

disc0 = ProtoplanetaryDisc_Zt(St=0.03)
disc1 = ProtoplanetaryDisc_Zrt(St=0.03)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), sharex=True, sharey=True)
font = 27

init_plot(ax, None, r'$\textnormal{r [AU]}$', 
          r'$Z$', font = font)
r_arr  = np.arange(1, 1000, 0.1)*u.AU
t_arr = np.array([0.2, 0.6, 1, 2, 3])*u.Myr
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

    

    ax.plot(r_arr,disc0.Z(r_arr, t)*np.ones(r_arr.shape),c=c, label=rf'${t.value:.{num}f}\,\mathrm{{Myr}}$',
             linewidth=4.5, alpha=alpha, zorder=dim-i, linestyle='-')
    ax.plot(r_arr,disc1.Z(r_arr, t).decompose(), c=c,
             linewidth=4.5, alpha=alpha, zorder=dim-i, linestyle='--')


ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.1, 1, 1), ncol=3, fontsize=font-4)

line1, = ax.plot([],[], c='gray', linewidth=4, label= r'$\textnormal{Constant St}$',linestyle='--')
line2, = ax.plot([],[], c='gray', linewidth=4, label=r'$\textnormal{Constant }\rm{St}_{\chi}$')

def fmt(x, pos):
    return rf'${x:.1f}$'

ax.set_xscale('log')

ax.set_yscale('log')

ax.set_xlim(5, 300)
ax.set_ylim(10**-6, 0.02)

handles = [line1, line2]
labels = [line.get_label() for line in handles]

# Create the legend above the subplots
fig.legend(handles, labels, loc='upper left',bbox_to_anchor=(0.18, 0., 1, 1),  ncol=3, fontsize=font-5.9)



plt.subplots_adjust(wspace=0.05, hspace=0.01, left=0.1, right=0.65, top=0.9, bottom=0.11)




fig.savefig('Z_comparison.pdf', dpi=500, format='pdf', bbox_inches='tight')
