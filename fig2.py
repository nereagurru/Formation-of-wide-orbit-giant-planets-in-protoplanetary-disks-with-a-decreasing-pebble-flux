#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:08:16 2025

@author: nerea
"""
from disc import ProtoplanetaryDisc_Zt as ProtoplanetaryDisc
import numpy as np
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt


vf = 1*u.m/u.s

def a_s(disc, St, r):
    return (St/(1*u.g/u.cm**3)*2/np.pi*disc.sigma_g(r, disc.t0.value, disc.t0.unit)).to(u.mm)


def St_drift(disc, r):
    return (np.sqrt(3*np.pi)/8*r*disc.kepler_angular(r)/disc.delta_v(r, 0.2*u.Myr)*0.1*disc.Z0).decompose()#0.03*(r/100/u.AU)**(-1/2)


def St_frag(disc, r):
    return 1/3/disc.delta*(vf/disc.cs(r))**2


fig, ax = plt.subplots(nrows=2, ncols=1,  sharex=True, figsize = (10, 10))


init_plot(ax[0], None, None, r'$\rm{St}$', font=28)
init_plot(ax[1], None, r'$\textnormal{r [AU]}$', r'$\mathrm{\tau_{g}\;[Myr]}$', font=28)


ax[0].secondary_xaxis('top')

line_dr = '-'
line_fr = '-'

disc = ProtoplanetaryDisc()
r_arr = np.arange(5, 400, 1)*u.AU

St_dr = St_drift(disc, r_arr).decompose()

St_fr = St_frag(disc, r_arr).decompose()

ax[0].plot(r_arr, St_fr, c='#03C988', label=r'$\rm{St}_{\rm{frag}}$', linewidth=3, zorder=1)

ax[0].plot(r_arr, St_dr, linestyle=line_dr, c='royalblue', 
           label=r'$\rm{St}_{\rm{drift}}$', linewidth=3, zorder=2)

ax[1].plot(r_arr, (np.log((a_s(disc, St_dr, r_arr)/u.um))*4/np.sqrt(3*np.pi)/0.01/disc.kepler_angular(r_arr)).to(u.Myr),
           linestyle=line_dr, c='royalblue', linewidth=3, zorder=2)
ax[1].plot(r_arr, (np.log((a_s(disc, St_fr, r_arr)/u.um))*4/np.sqrt(3*np.pi)/0.01/disc.kepler_angular(r_arr)).to(u.Myr),
           c='#03C988', linewidth=3, zorder=1)




ax[0].plot(r_arr, 0.03*u.mm*np.ones(r_arr.shape), c='lightgrey', linestyle=':', linewidth=2.5, zorder=0)
ax[0].plot(r_arr, disc.St(r_arr, 0.2*u.Myr), c='r', linestyle=line_dr, zorder=0, alpha=0.4)
ax[0].axvline(x=100, color='lightgray', linestyle=':', alpha=0.5, zorder=1, linewidth=2.5)
ax[0].text(120, 0.032, r'$\rm{St}\sim 0.03$', c='grey', fontsize=25)

ax[0].text(7, 0.043, r'$\rm{St}_\chi\;\rm{const}$', c='r', fontsize=20, alpha=0.4)
ax[1].plot(r_arr, 0.2*np.ones(r_arr.shape), c='lightgrey', linestyle=':', zorder=1, linewidth=2.5)
ax[1].axvline(x=100, color='lightgray', linestyle=':', alpha=0.5, zorder=1, linewidth=2.5)
ax[1].text(36, 0.25, r'$\rm{\tau}_{\rm{g}}\sim 0.2\,\mathrm{Myr}$', c='grey', fontsize=25)


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_ylim(0.01, 0.11)



ax[0].set_yticks([0.01, 0.1])
ax[0].legend(loc='lower center', ncol=2, fontsize=22)
ax[0].set_yticks([0.01,0.1])

ax[1].set_yticks([0.01,0.1])

ax[0].set_xlim(5, 300)



ax2 = ax[1].twinx()
ax2.minorticks_on()
ax2.tick_params(direction='in', length=8, width=1.75,
            which = 'major', top=True, right=True, pad=7)
ax2.tick_params(direction='in', length=9, width=1.75,
            which = 'minor', top=True, right=True)
ax2.set_ylabel(r'$\rm{a_{s}}\,\mathrm{[mm]}$')

ax2.plot(r_arr, a_s(disc, St_dr, r_arr).to(u.mm),
           linestyle='--', c='royalblue', linewidth=3, zorder=2, alpha=0.4)
ax2.plot(r_arr, a_s(disc, St_fr, r_arr).to(u.mm),
           c='#03C988', linewidth=3, zorder=1, alpha=0.4, linestyle='--')
ax2.set_yscale('log')
ax2.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
ax[1].plot([], [], linestyle='-', label=r'$\mathrm{\tau_{g}}$', c='k', linewidth=3)
ax[1].plot([], [], linestyle='--', label=r'$\mathrm{a_{s}}$', c='k', linewidth=3)
ax[1].legend(loc='lower center', ncol=2, fontsize=22)
plt.subplots_adjust(wspace=0.0, hspace=0.1, left=0.2, right=0.85, top=0.90, bottom=0.2)
fig.savefig('St_tau_lim.pdf', dpi=500, format='pdf', bbox_inches='tight')
