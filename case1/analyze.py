import os
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

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
COLORS  = mpl.cm.get_cmap('jet')
DATES   = pd.date_range('2020-08-01', '2025-01-01', freq='MS')

def get_iterations(folder, filekey):
    files_in_dir = os.listdir(folder)
    debug_files  = [name for name in files_in_dir if 'optimize_result' in name]
    iterations   = len(debug_files)

    values = []
    for iter in range(iterations):
        info = np.load(folder + f'/optimize_result_{iter}.npz')

        try:
            values.append(info[filekey])
        except:
            raise ValueError(f'{filekey} not in npz. The possible keys are: {info.files}')
    
    return np.asarray(values)

def plot_obj(wind, gas):

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(9, 7))
    
    for w, weight in enumerate(WEIGHTS):
        
        ax[0].plot(wind['F'][w],
                   color=COLORS(weight))
        ax[0].axhline(y=wind['Fev'][w].mean(),
                      ls='--',
                      color=COLORS(weight),
                      zorder=0,
                      lw=1)


        ax[1].plot(gas['F'][w],
                   color=COLORS(weight),
                   label=rf'$\omega={weight}$')
        ax[1].axhline(y=gas['Fev'][w],
                      ls='--',
                      color=COLORS(weight),
                      zorder=0,
                      lw=1)

    ax[0].set_title('wind + gas')
    ax[1].set_title('only gas')
    ax[0].set_ylabel(r'$F=\omega f_1 + (1-\omega)f_2$')
    ax[0].set_xlabel('iteration')
    ax[0].set_xlim(-0.5,21)
    ax[1].set_xlim(-0.5,21)
    ax[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax[1].legend(loc=(1.01, 0.075), fontsize=11)


    plt.tight_layout()
    plt.show()

def plot_controls(wind, gas):

    qMax, pMax = 8000, 18_000
    nSteps = 54
    w = 6

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='row', figsize=(10/1.5, 6/1.5))
    fig.autofmt_xdate(ha='center')
    

    ax[0,0].step(DATES,
                 qMax*wind['rates'][w][-1,:nSteps],
                 where='mid',
                 color=COLORS(0.2))
    ax[0,1].step(DATES,
                 qMax*gas['rates'][w][-1,:nSteps],
                 where='mid',
                 color=COLORS(0.2))
    

    ax[1,0].step(DATES,
                 qMax*wind['rates'][w][-1,nSteps:2*nSteps],
                 where='mid',
                 color=COLORS(0.2))
    ax[1,1].step(DATES,
                 qMax*gas['rates'][w][-1,nSteps:2*nSteps],
                 where='mid',
                 color=COLORS(0.2))
    

    ax[2,0].step(DATES,
                 pMax*wind['rates'][w][-1,-nSteps:],
                 where='mid',
                 color=COLORS(0.2))
    ax[2,1].step(DATES,
                 pMax*gas['rates'][w][-1,-nSteps:],
                 where='mid',
                 color=COLORS(0.2))
    

     # labels and titles
    ax[0,0].set_title('wind + gas')
    ax[0,1].set_title('only gas')

    #ax[0,0].set_ylabel(r'Injrate A5 [Sm$^3$/day]', fontsize=11)
    #ax[1,0].set_ylabel(r'Injrate A6 [Sm$^3$/day]', fontsize=11)
    #ax[2,0].set_ylabel(r'Prodrate OP [Sm$^3$/day]', fontsize=11)

    ax[0,0].set_ylim(-200, 10000)
    ax[1,0].set_ylim(-200, 10000)
    ax[2,0].set_ylim(-500, 20500)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.05)
    plt.show()


if __name__ == '__main__':

    data_wind = {'npv': [], 'iem': [], 'em': [], 'F': [], 'rates': [], 'Fev': []}
    data_gas  = {'npv': [], 'iem': [], 'em': [], 'F': [], 'rates': [], 'Fev': []}

    for weight in WEIGHTS:
        file_gas = np.load(f'pareto_nowind/weight_{weight}/final_run.npz')
        path_gas = f'pareto_nowind/weight_{weight}'

        file1 = np.load(f'pareto/weight_{weight}/final_run.npz')
        file2 = np.load(f'pareto2/weight_{weight}/final_run.npz')
        if file1['obj'].mean() < file1['obj'].mean():
            file_wind = file1
            path_wind = f'pareto/weight_{weight}'
        else:
            file_wind = file2
            path_wind = f'pareto2/weight_{weight}'

        # add data
        data_wind['npv'].append(file_wind['npv']/1e9)                       # Billion usd
        data_wind['iem'].append(file_wind['intensity'])                     # kg/toe
        data_wind['em'].append(file_wind['emissions'].sum(axis=1)/1000)     # kilo tonnes
        data_wind['F'].append(get_iterations(path_wind, 'fun'))
        data_wind['Fev'].append(file_wind['obj'])
        data_wind['rates'].append(get_iterations(path_wind, 'x'))

        data_gas['npv'].append(file_gas['npv'].mean()/1e9)
        data_gas['iem'].append(file_gas['intensity'].mean())
        data_gas['em'].append(file_gas['emissions'].sum()/1000)
        data_gas['F'].append(get_iterations(path_gas, 'fun'))
        data_gas['Fev'].append(file_gas['obj'])
        data_gas['rates'].append(get_iterations(path_gas, 'x'))
    

    #plot_obj(data_wind, data_gas)   
    plot_controls(data_wind, data_gas)