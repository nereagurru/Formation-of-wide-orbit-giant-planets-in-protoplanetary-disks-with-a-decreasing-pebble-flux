from disc import ProtoplanetaryDisc_constZ
from disc import ProtoplanetaryDisc_Zt
from disc import ProtoplanetaryDisc_Zrt
import numpy as np
from protoplanet import Protoplanet
import astropy.units as u
from Plotting import init_plot
import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(150, 120))

tf = 5*u.Myr


M0 = 0.01*u.Mearth
color = ['#2F58CD','dodgerblue','#97DEFF']

for i, St in enumerate([0.01, 0.03, 0.06]):
    disc_arr = [ProtoplanetaryDisc_constZ(St=St),
                ProtoplanetaryDisc_Zt(St=St),
                ProtoplanetaryDisc_Zrt(St=St)]
    for j, disc in enumerate(disc_arr):
        labelx = r'$\mathrm{r\,[AU]}$' if j == 2 else None
        labely = r'$\mathrm{M\,[M_{\oplus}]}$' if i == 0 else None
        title = rf'$\rm{{St}}={St}$' if j==0 else None
        init_plot(ax[j, i], title, labelx, labely, font=22)
        for r0 in [20, 50, 80]*u.AU:
            print(r0)
            for t0, order, c, line in zip([0.2, 0.5, 0.8]*u.Myr, [1, 2, 3],
                                    color, [':', '--', '-']):
                protoplanet = Protoplanet(disc, r0)
                ts, rs, Ms, tg, rg, Mg = protoplanet.rM_numerical(t0, tf, max_step=0.005)
                ax[j, i].plot(rs, Ms, c=c,linewidth=2.5, zorder=order,
                              linestyle=line)
                if tg!=None:
                    ax[j, i].plot(rg, Mg)

        r_iso = np.arange(0.8, 85, 1)*u.AU
        ax[j,i].plot(r_iso, protoplanet.Miso(r_iso), c='gray', linestyle='-.', alpha=0.5)



line1, = ax[0,0].plot([],[], linestyle=':',  c=color[0], linewidth=2.5, label=r'$\mathrm{t_{0, p}=0.2\,\mathrm{Myr}}$')
line2, = ax[0,0].plot([],[], linestyle='--', c=color[1], linewidth=2.5, label=r'$\mathrm{t_{0, p}=0.5\,\mathrm{Myr}}$')
line3, = ax[0,0].plot([],[], c=color[2], linewidth=2.5, label=r'$\mathrm{t_{0, p}=0.8\,\mathrm{Myr}}$')


ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_xlim(7, 95)

ax[0, 0].set_yticks([0.1, 10])

for axx in ax.flatten():
    # Reduce the linewidth of the axes
    axx.spines['top'].set_linewidth(1.5)
    axx.spines['bottom'].set_linewidth(1.5)
    axx.spines['left'].set_linewidth(1.5)
    axx.spines['right'].set_linewidth(1.5)
#fig.tight_layout()

plt.subplots_adjust(wspace=0.08, hspace=0.15, left=0.18, right=0.95, top=0.83)  
handles = [line1, line2, line3]
labels = [line.get_label() for line in handles]

# Create the legend above the subplots
fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.05, 0., 1, 1),  ncol=3, fontsize=20)


# Adjust the layout to make room for the legend

fig.text(0.05, 0.7, r'$\mathrm{(1)}\,Z_{0}$', transform=fig.transFigure,
         rotation=0, ha='center', va='center')

fig.text(0.05, 0.47, r'$\mathrm{(2)}\,Z(t)$', transform=fig.transFigure,
         rotation=0, ha='center', va='center')

fig.text(0.05, 0.23, r'$\mathrm{(3)}\,Z(r, t)$', transform=fig.transFigure,
         rotation=0, ha='center', va='center')

fig.savefig('FluxModel_comparison.pdf', dpi=500, format='pdf', bbox_inches='tight')
