import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mplcyberpunk

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


def plot_pareto_front():

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    npv = []
    iem = []

    npv_ref = []
    iem_ref = []
    em_ref = []

    for w in weights:
        
        # choose best value
        data1 = np.load(f'pareto_wind_run1/weight_{w}/final_run.npz')
        data2 = np.load(f'pareto_wind_run2/weight_{w}/final_run.npz')

        obj1 = data1['obj'].mean()
        obj2 = data2['obj'].mean()
        
        if obj1 < obj2:
            data = data1
        else:
            data = data2

        npv.append(data['npv']/1e9)
        iem.append(data['intensity'])

        # no wind
        data_ref = np.load(f'pareto_nowind/weight_{w}/final_run.npz')
        npv_ref.append(np.mean(data_ref['npv']/1e9))
        iem_ref.append(np.mean(data_ref['intensity']))
        em_ref.append(np.sum(data_ref['emissions']/1000))

    # plot 
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))
    colors = mpl.cm.get_cmap('jet')

    for w, weight in enumerate(weights):

        ax[0].scatter(iem[w].mean(), npv[w].mean(), color=colors(weight), zorder=2, ec='gray', lw=0.5, s=45)
        ax[0].errorbar(iem[w].mean(),
                       npv[w].mean(), 
                       xerr=[[iem[w].mean()-iem[w].min()], 
                             [iem[w].max()]-iem[w].mean()], 
                       yerr=[[npv[w].mean()-npv[w].min()], 
                             [npv[w].max()]-npv[w].mean()], 
                       color=colors(weight), 
                       fmt='o', 
                       markersize=0.0, 
                       zorder=1, 
                       alpha=0.9, 
                       elinewidth=1.75)
    
    ax[0].set_ylabel('NPV [Billion USD]')
    ax[0].set_xlabel('Emission intensity [kg co2/toe]')
    ax[0].set_title('wind + gas')
    
    coefficients = np.polyfit(np.array(iem).mean(1), np.array(npv).mean(1), deg=2)
    fit = np.poly1d(coefficients)

    x_fit = np.linspace(min(np.array(iem).mean(1)), max(np.array(iem).mean(1)), 100)
    y_fit = fit(x_fit)
    ax[0].plot(x_fit, y_fit, color='gray', zorder=0, ls='--', label=r'polyfit', alpha=0.8)
    ax[0].legend(loc='lower right', frameon=True, framealpha=1)

    # second ax
    for w, weight in enumerate(weights):
        ax[1].scatter(iem_ref[w], npv_ref[w], color=colors(weight), zorder=2, label=rf'$\omega={weight}$', ec='gray', lw=0.5, s=45)

    coefficients = np.polyfit(iem_ref, npv_ref, deg=2)
    fit = np.poly1d(coefficients)
    x_fit = np.linspace(min(iem_ref), max(iem_ref), 100)
    ax[1].plot(x_fit, fit(x_fit), color='gray', zorder=0, ls='--', alpha=0.8)

    ax[1].set_xlabel('Emission intensity [kg co2/toe]')
    ax[1].legend(loc=(1.01, 0.075), fontsize=11)
    ax[1].set_title('only gas')

    plt.tight_layout()
    plt.draw()
    fig.savefig('../figures/case1_intensity_vs_npv.pdf')


def plot_pareto_npv_vs_em():

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    npv = []
    em = []

    npv_ref = []
    em_ref = []

    for w in weights:
        
        # choose best value
        data1 = np.load(f'pareto_wind_run1/weight_{w}/final_run.npz')
        data2 = np.load(f'pareto_wind_run2/weight_{w}/final_run.npz')

        obj1 = data1['obj'].mean()
        obj2 = data2['obj'].mean()
        
        if obj1 < obj2:
            data = data1
        else:
            data = data2

        npv.append(data['npv']/1e9)
        em.append(np.sum(data['emissions']/1000, axis=1))

        # no wind
        data_ref = np.load(f'pareto_nowind/weight_{w}/final_run.npz')
        npv_ref.append(data_ref['npv']/1e9)
        em_ref.append(np.sum(data_ref['emissions']/1000))

    # plot 
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))
    colors = mpl.cm.get_cmap('jet')

    for w, weight in enumerate(weights):

        ax[0].scatter(em[w].mean(), npv[w].mean(), color=colors(weight), zorder=2, ec='gray', lw=0.5, s=45)
        ax[0].errorbar(em[w].mean(), 
                       npv[w].mean(), 
                       xerr=[[em[w].mean()-em[w].min()], 
                             [em[w].max()]-em[w].mean()], 
                       yerr=[[npv[w].mean()-npv[w].min()], 
                             [npv[w].max()]-npv[w].mean()], 
                       color=colors(weight), 
                       fmt='o', 
                       markersize=0.0, 
                       zorder=1, 
                       alpha=0.9, 
                       elinewidth=1.75)

    ax[0].set_ylabel('NPV [Billion USD]')
    ax[0].set_xlabel('Emissions [kilotonnes]')
    ax[0].set_title('wind + gas')


    # second ax
    for w, weight in enumerate(weights):
        ax[1].scatter(em_ref[w], npv_ref[w], color=colors(weight), zorder=2, label=rf'$\omega={weight}$', ec='gray', lw=0.5, s=45)

    ax[1].set_xlabel('Emissions [kilotonnes]')
    ax[1].legend(loc=(1.01, 0.075), fontsize=11)
    ax[1].set_title('only gas')

    plt.tight_layout()
    plt.draw()
    fig.savefig('../figures/case1_emissions_vs_npv.pdf')



def plot_prod_data():

    folder1 = 'pareto'
    folder2 = 'pareto2'

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dates = pd.date_range('2020-08-01', '2025-01-01', freq='MS')

    fopt = [[] for w in weights]
    fgpt = [[] for w in weights]
    fwit = [[] for w in weights]
    
    for w, weight in enumerate(weights):

        # choose best value
        data1 = np.load(f'{folder1}/weight_{weight}/final_run.npz', allow_pickle=True)
        data2 = np.load(f'{folder2}/weight_{weight}/final_run.npz', allow_pickle=True)

        obj1 = data1['obj'].mean()
        obj2 = data2['obj'].mean()
        
        if obj1 < obj2:
            pred_data = data1['pred_data']
        else:
            pred_data = data2['pred_data']


        for data in pred_data:
            
            fopt[w].append(data['fopt'].squeeze())
            fgpt[w].append(data['fgpt'].squeeze())
            fwit[w].append(data['fwit'].squeeze())
    
    # plot
    fig, ax = plt.subplots(ncols=3, figsize=(12, 3.5), sharex=True)
    fig.autofmt_xdate(ha='center')

    colors = sns.color_palette('seismic', n_colors=len(weights))
    colors[5] = 'lightgrey'

    for w, weight in enumerate(weights):
        ax[0].plot(dates, fopt[w], color=colors[w])
        ax[1].plot(dates, fgpt[w], color=colors[w])
        ax[2].plot(dates, fwit[w], color=colors[w], label=rf'$\omega={weight}$')
    
    ax[0].set_title('Total oil production [Sm³]', fontsize=14)
    ax[0].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%Y'))

    ax[1].set_title('Total gas production [Sm³]', fontsize=14)
    ax[1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%Y'))

    ax[2].set_title('Total water injection [Sm³]', fontsize=14)
    ax[2].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%Y'))
    ax[2].legend(loc=(1.05, -0.075), fontsize=11)

    plt.tight_layout()
    plt.draw()
    #fig.savefig('prod.pdf')






if __name__ == '__main__':

    #plot_pareto_front()
    plot_pareto_npv_vs_em()
    #plot_prod_data()
    plt.show()

