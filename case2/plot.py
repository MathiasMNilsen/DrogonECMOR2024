import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mplcyberpunk
import os

plt.style.use("cyberpunk")
mpl.rcParams.update({'text.color' : 'black',
                     'axes.labelcolor' : 'black',
                     'xtick.color' : 'black',
                     'ytick.color' : 'black',
                     'grid.linestyle' : '-',
                     'grid.color' : 'gray',
                     'grid.alpha' : 0.4,
                     'figure.facecolor' : 'white',
                     'axes.facecolor' : 'white',
                     'savefig.facecolor': 'white'})

COLOR_MAP = 'jet'#'Spectral'
DATES = pd.date_range('2020-08-01', '2025-01-01', freq='MS')


def pareto_front():

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    data    = {'npv': [], 'iem': [], 'em': []}
    data_nw = {'npv': [], 'iem': [], 'em': []}

    for weight in weights:

        datafile = np.load(f'pareto_wind/weight_{weight}/eval_50.npz')
        data['npv'].append(datafile['npv']/1e9)
        data['iem'].append(datafile['intensity'])
        data['em'].append(datafile['emissions'].sum(axis=1)/1000)

        datafile_nw = np.load(f'pareto_nowind/weight_{weight}/eval.npz')
        data_nw['npv'].append(datafile_nw['npv']/1e9)
        data_nw['iem'].append(datafile_nw['intensity'])
        data_nw['em'].append(datafile_nw['emissions'].sum()/1000)

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    colors = mpl.cm.get_cmap(COLOR_MAP)

    for w, weight in enumerate(weights):

        # mean with wind
        ax[0].scatter(data['em'][w].mean(),
                      data['npv'][w].mean(),
                      color=colors(weight),
                      zorder=2, 
                      ec='gray', 
                      lw=0.5, 
                      s=40)
        
        # std with wind
        ax[0].errorbar(data['em'][w].mean(),
                       data['npv'][w].mean(), 
                       xerr=[[data['em'][w].mean()-data['em'][w].min()], 
                             [data['em'][w].max()]-data['em'][w].mean()], 
                       yerr=[[data['npv'][w].mean()-data['npv'][w].min()], 
                             [data['npv'][w].max()]-data['npv'][w].mean()], 
                       color=colors(weight), 
                       fmt='o', 
                       markersize=0.0, 
                       zorder=1, 
                       alpha=0.9, 
                       elinewidth=1.75)
        
        # no wind 
        ax[1].scatter(data_nw['em'][w],
                      data_nw['npv'][w],
                      color=colors(weight),
                      label=rf'$\omega={weight}$',
                      zorder=2, 
                      ec='gray', 
                      lw=0.5, 
                      s=40)
        
    # curve fit 1
    fit_1 = np.poly1d(np.polyfit(np.array(data['em']).mean(axis=1),
                                 np.array(data['npv']).mean(axis=1), 
                                 deg=3))
    x_fit_1 = np.linspace(min(np.array(data['em']).mean(axis=1)), 
                            max(np.array(data['em']).mean(axis=1)), 
                            100)        
    ax[0].plot(x_fit_1, 
               fit_1(x_fit_1), 
               color='gray', 
               zorder=0, 
               ls='--', 
               label='polyfit')
    

    # curve fit 2
    data_nw['em']  = np.array(data_nw['em']) 
    data_nw['npv'] = np.array(data_nw['npv']).squeeze()
    fit_2 = np.poly1d(np.polyfit(data_nw['em'],
                                 data_nw['npv'], 
                                 deg=3))
    x_fit_2 = np.linspace(min(data_nw['em']), 
                          max(data_nw['em']), 
                          100)        
    ax[1].plot(x_fit_2, 
               fit_2(x_fit_2), 
               color='gray', 
               zorder=0, 
               ls='--')
    
    
    # labels and stuff
    ax[0].set_ylabel('NPV [Billion USD]')
    ax[0].set_xlabel('Emissions [kilotonnes]')
    ax[0].set_title('wind + gas')
    ax[0].legend(loc='lower right', frameon=True, framealpha=1)

    ax[1].legend(loc=(1.01, 0.075), fontsize=11)
    ax[1].set_xlabel('Emissions [kilotonnes]')
    ax[1].set_title('only gas')
        
    plt.tight_layout()
    plt.draw()

    fig.savefig('../figures/case2_emissions_vs_npv.pdf')


def controls():
    from parametrization import log_poly

    qMax, pMax = 8000, 18_000
    al, au = -5, 5
    bl, bu = -10, 10
    cl, cu = -20, 20

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='row', figsize=(10, 6))
    fig.autofmt_xdate(ha='center')
    colors  = mpl.cm.get_cmap(COLOR_MAP)

    for weight in weights:

        ##################### WIND ###################
        folder = f'pareto_wind/weight_{weight}'
        last_iter = len([f for f in os.listdir(folder) if 'optimize_result' in f]) - 1
        x = np.load(f'{folder}/optimize_result_{last_iter}.npz')['x']
        a = al + x[:3]*(au-al)
        b = bl + x[3:6]*(bu-bl)
        c = cl + x[6:]*(cu-cl)

        qA5_w = log_poly(a[0], b[0], c[0], min=0, max=qMax, steps=54)
        qA6_w = log_poly(a[1], b[1], c[1], min=0, max=qMax, steps=54)
        qOP_w = log_poly(a[2], b[2], c[2], min=0, max=pMax, steps=54)

        ax[0,0].plot(DATES,
                     qA5_w,
                     color=colors(weight))
        
        ax[1,0].plot(DATES,
                     qA6_w,
                     color=colors(weight))
        
        ax[2,0].plot(DATES,
                     qOP_w,
                     color=colors(weight))
    

        ##################### No WIND ###################
        folder = f'pareto_nowind/weight_{weight}'
        last_iter = len([f for f in os.listdir(folder) if 'optimize_result' in f]) - 1
        x = np.load(f'{folder}/optimize_result_{last_iter}.npz')['x']
        a = al + x[:3]*(au-al)
        b = bl + x[3:6]*(bu-bl)
        c = cl + x[6:]*(cu-cl)

        qA5_nw = log_poly(a[0], b[0], c[0], min=0, max=qMax, steps=54)
        qA6_nw = log_poly(a[1], b[1], c[1], min=0, max=qMax, steps=54)
        qOP_nw = log_poly(a[2], b[2], c[2], min=0, max=pMax, steps=54)

        ax[0,1].plot(DATES,
                     qA5_nw,
                     color=colors(weight))
        
        ax[1,1].plot(DATES,
                     qA6_nw,
                     color=colors(weight),
                     label=rf'$\omega={weight}$')
        
        ax[2,1].plot(DATES,
                     qOP_nw,
                     color=colors(weight))
    

    # labels and titles
    ax[0,0].set_title('wind + gas')
    ax[0,1].set_title('only gas')
    ax[1,1].legend(loc=(1.01, -0.25), fontsize=11, handlelength=1)

    ax[0,0].set_ylabel(r'Injrate A5 [Sm$^3$/day]', fontsize=11)
    ax[1,0].set_ylabel(r'Injrate A6 [Sm$^3$/day]', fontsize=11)
    ax[2,0].set_ylabel(r'Prodrate OP [Sm$^3$/day]', fontsize=11)

    ax[0,0].set_ylim(-200, 8200)
    ax[1,0].set_ylim(-200, 8200)
    ax[2,0].set_ylim(-500, 20500)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.draw()

    fig.savefig('../figures/case2_controls.pdf')


def production():

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey='row', figsize=(10, 12))
    fig.autofmt_xdate(ha='center')
    colors  = mpl.cm.get_cmap(COLOR_MAP)

    for w, weight in enumerate(weights):

        ##################### WIND ###################
        file_w = f'pareto_wind/weight_{weight}/eval_50.npz'
        prod_data_w = np.load(file_w, allow_pickle=True)['pred_data']
        emissions_w = np.load(file_w, allow_pickle=True)['emissions'].mean(axis=0)/1000
        emissions_w = np.insert(emissions_w, 0, 0)

        fwpt_w = []
        fgpt_w = []
        fopt_w = []

        for data in prod_data_w:
            fwpt_w.append(data['fwpt'].squeeze())
            fgpt_w.append(data['fgpt'].squeeze())
            fopt_w.append(data['fopt'].squeeze())

        
        ax[0,0].plot(DATES,
                     fopt_w,
                     color=colors(weight),
                     zorder=len(weights)-w)
        
        ax[1,0].plot(DATES,
                     fgpt_w,
                     color=colors(weight),
                     zorder=len(weights)-w)
        
        ax[2,0].plot(DATES,
                     fwpt_w,
                     color=colors(weight),
                     zorder=len(weights)-w)
        
        ax[3,0].plot(DATES,
                     np.cumsum(emissions_w),
                     color=colors(weight),
                     zorder=len(weights)-w)
        

        ##################### NO WIND ###################
        file_nw = f'pareto_nowind/weight_{weight}/eval.npz'
        prod_data_nw = np.load(file_nw, allow_pickle=True)['pred_data']
        emissions_nw = np.load(file_nw, allow_pickle=True)['emissions'].squeeze()/1000
        emissions_nw = np.insert(emissions_nw, 0, 0)

        fwpt_nw = []
        fgpt_nw = []
        fopt_nw = []

        for data in prod_data_nw:
            fwpt_nw.append(data['fwpt'].squeeze())
            fgpt_nw.append(data['fgpt'].squeeze())
            fopt_nw.append(data['fopt'].squeeze())


        ax[0,1].plot(DATES,
                     fopt_nw,
                     color=colors(weight),
                     zorder=len(weights)-w)
        
        ax[1,1].plot(DATES,
                     fgpt_nw,
                     color=colors(weight),
                     label=rf'${weight}$',
                     zorder=len(weights)-w)
        
        ax[2,1].plot(DATES,
                     fwpt_nw,
                     color=colors(weight),
                     zorder=len(weights)-w)
        
        ax[3,1].plot(DATES,
                     np.cumsum(emissions_nw),
                     color=colors(weight),
                     zorder=len(weights)-w)

    # labels and titles
    ax[0,0].set_title('wind + gas')
    ax[0,1].set_title('only gas')
    ax[1,1].legend(loc=(1.01, -0.8), fontsize=11, handlelength=1)

    ax[0,0].set_ylabel(r'FOPT [Sm$^3$]', fontsize=11)
    ax[1,0].set_ylabel(r'FGPT [Sm$^3$]', fontsize=11)
    ax[2,0].set_ylabel(r'FWPT [Sm$^3$]', fontsize=11)
    ax[3,0].set_ylabel(r'Avg. Emissions [kilotonnes]', fontsize=11)

    #norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    #boundarynorm = mpl.colors.BoundaryNorm(boundaries=weights, ncolors=colors.N)
    #sm = plt.cm.ScalarMappable(cmap=colors) 
    #cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
    #cbar = plt.colorbar(sm, ax=ax[1,1], cax=cbar_ax, ticks=weights) 
    #cbar.ax.locator_params(axis='y', nbins=len(weights)+1)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.draw()

    fig.savefig('../figures/case2_production.pdf')


def pareto_front_intensity():

    from scipy.optimize import curve_fit

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    data    = {'npv': [], 'iem': [], 'em': []}
    data_nw = {'npv': [], 'iem': [], 'em': []}

    for weight in weights:

        datafile = np.load(f'pareto_wind/weight_{weight}/eval_50.npz')
        data['npv'].append(datafile['npv']/1e9)
        data['iem'].append(datafile['intensity'])
        data['em'].append(datafile['emissions'].sum(axis=1)/1000)

        datafile_nw = np.load(f'pareto_nowind/weight_{weight}/eval.npz')
        data_nw['npv'].append(datafile_nw['npv']/1e9)
        data_nw['iem'].append(datafile_nw['intensity'])
        data_nw['em'].append(datafile_nw['emissions'].sum()/1000)

    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(9, 4))
    colors = mpl.cm.get_cmap(COLOR_MAP)

    # zoom in axis
    x1, x2, y1, y2 = 25, 35, 28, 30  # subregion of the original image
    ax_zoom1 = ax[0].inset_axes(bounds=[0.25, 0.55, 0.35, 0.35], 
                                xlim=(x1, x2), 
                                ylim=(y1-0.2, y2), 
                                xticklabels=[], 
                                yticklabels=[],
                                yticks=[],
                                zorder=2
                                )
    ax_zoom2 = ax[1].inset_axes(bounds=[0.35, 0.55, 0.35, 0.35], 
                                xlim=(x1+12, x2+18), 
                                ylim=(y1-2, y2-0.5), 
                                xticklabels=[], 
                                yticklabels=[],
                                yticks=[],
                                zorder=2
                                )

    for w, weight in enumerate(weights):

        # mean with wind
        ax[0].scatter(data['iem'][w].mean(),
                      data['npv'][w].mean(),
                      color=colors(weight),
                      zorder=2, 
                      ec='gray', 
                      lw=0.5, 
                      s=40)
        
        # std with wind
        ax[0].errorbar(data['iem'][w].mean(),
                       data['npv'][w].mean(), 
                       xerr=[[data['iem'][w].mean()-data['iem'][w].min()], 
                             [data['iem'][w].max()]-data['iem'][w].mean()], 
                       yerr=[[data['npv'][w].mean()-data['npv'][w].min()], 
                             [data['npv'][w].max()]-data['npv'][w].mean()], 
                       color=colors(weight), 
                       fmt='o', 
                       markersize=0.0, 
                       zorder=1, 
                       alpha=0.9, 
                       elinewidth=1.75)
        
        # no wind 
        ax[1].scatter(data_nw['iem'][w],
                      data_nw['npv'][w],
                      color=colors(weight),
                      label=rf'$\omega={weight}$',
                      zorder=2, 
                      ec='gray', 
                      lw=0.5, 
                      s=40)
    
        # zoom data
        ax_zoom1.scatter(data['iem'][w].mean(),
                        data['npv'][w].mean(),
                        color=colors(weight),
                        zorder=2, 
                        ec='gray', 
                        lw=0.5, 
                        s=40)
        ax_zoom1.errorbar(data['iem'][w].mean(),
                       data['npv'][w].mean(), 
                       xerr=[[data['iem'][w].mean()-data['iem'][w].min()], 
                             [data['iem'][w].max()]-data['iem'][w].mean()], 
                       yerr=[[data['npv'][w].mean()-data['npv'][w].min()], 
                             [data['npv'][w].max()]-data['npv'][w].mean()], 
                       color=colors(weight), 
                       fmt='o', 
                       markersize=0.0, 
                       zorder=1, 
                       alpha=0.9, 
                       elinewidth=1.75)
        
        ax_zoom2.scatter(data_nw['iem'][w],
                      data_nw['npv'][w],
                      color=colors(weight),
                      label=rf'$\omega={weight}$',
                      zorder=2, 
                      ec='gray', 
                      lw=0.5, 
                      s=40)
              
    ax[0].indicate_inset_zoom(ax_zoom1, edgecolor='blue', lw=0.75)
    ax[1].indicate_inset_zoom(ax_zoom2, edgecolor='blue', lw=0.75)
    rect1 = mpl.patches.Rectangle((0.25, 0.55), 0.35, 0.35, 
                                 linewidth=0.75, 
                                 edgecolor='blue', 
                                 facecolor='none',
                                 transform=ax[0].transAxes,
                                 zorder=3)
    rect2 = mpl.patches.Rectangle((0.35, 0.55), 0.35, 0.35, 
                                 linewidth=0.75, 
                                 edgecolor='blue', 
                                 facecolor='none',
                                 transform=ax[1].transAxes,
                                 zorder=3)
    ax[0].add_patch(rect1)
    ax[1].add_patch(rect2)

    data['iem'] = np.array(data['iem']).squeeze()
    data['npv'] = np.array(data['npv']).squeeze()

    data_nw['iem'] = np.array(data_nw['iem']).squeeze()
    data_nw['npv'] = np.array(data_nw['npv']).squeeze()

    ax[0].plot(data['iem'].mean(axis=1), 
               data['npv'].mean(axis=1), 
               '--', 
               color='steelblue', 
               zorder=0,
               lw=1.25)
    
    ax[1].plot(data_nw['iem'], 
               data_nw['npv'], 
               '--', 
               color='steelblue', 
               zorder=0,
               lw=1.25)
    
    ax_zoom1.plot(data['iem'].mean(axis=1), 
               data['npv'].mean(axis=1), 
               '--', 
               color='steelblue', 
               zorder=0,
               lw=0.75)

    ax_zoom2.plot(data_nw['iem'], 
               data_nw['npv'], 
               '--', 
               color='steelblue', 
               zorder=0,
               lw=0.75)

    '''
    def func(x, a, b, c):
        return a*np.exp(-b*x) + c

    # curve fit 1
    popt_1, _ = curve_fit(func,   
                          data['npv'].mean(axis=1),
                          data['iem'].mean(axis=1), 
                          p0=[10, 0.01, 0],
                          sigma=data['iem'].mean(axis=1))
    x_fit_1 = np.logspace(np.log10(min(np.array(data['npv']).mean(axis=1))), 
                          np.log10(max(np.array(data['npv']).mean(axis=1))), 
                          100)    
    ax[0].plot(func(x_fit_1, *popt_1),
               x_fit_1, 
               color='gray', 
               zorder=0, 
               ls='--', 
               label='exp-fit')

    # curve fit 2

    popt_2, _ = curve_fit(func,  
                          data_nw['iem'],  
                          data_nw['npv'],
                          p0=[10, 0.01, 0])
    x_fit_2 = np.logspace(np.log10(min(data_nw['iem'])), 
                          np.log10(max(data_nw['iem'])), 
                          100)        
    ax[1].plot(x_fit_2, 
               func(x_fit_2, *popt_2), 
               color='gray', 
               zorder=0, 
               ls='--')
    '''
    
    # labels and stuff
    ax[0].set_ylabel('NPV [Billion USD]')
    ax[0].set_xlabel('Emissions Intensity [kg co2/toe]')
    ax[0].set_title('wind + gas')
    #ax[0].legend(loc=(0.74, 0.1), frameon=False, framealpha=1)
    ax[0].set_xscale('log')
    ax[0].grid(True, which='both')

    ax[1].legend(loc=(1.01, 0.075), fontsize=11)
    ax[1].set_xlabel('Emissions Intensity [kg co2/toe]')
    ax[1].set_title('only gas')
    ax[1].set_xscale('log')
    ax[1].set_xticks([20, 30, 50 ,100, 500, 1000, 2000])
    ax[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax[1].grid(True, which='both')
        
    plt.tight_layout()
    plt.draw()

    fig.savefig('../figures/case2_intensity_vs_npv.pdf')


if __name__ == '__main__':

    pareto_front()
    #controls()
    production()
    #pareto_front_intensity()
    plt.show()