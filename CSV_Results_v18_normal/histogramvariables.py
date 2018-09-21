#!/bin/python
"""
    A script that plots and individual variable, showing signal/background seperation
    """
# Authors: Patrick Greenway

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
}

nJets = 2
variables = variables_map[nJets]
variables = ['MV1cB1']

def main():
    #df1 = pd.read_csv('allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    #df2 = pd.read_csv('allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    
    df1 = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    df2 = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    
    df = df1.append(df2)
    df = df.reset_index(drop=True)
    
    
    for variable in variables:
        values = df[variable]
        print values
        min_val = np.percentile(values, 25)
        max_val = np.percentile(values, 100)
        bins = np.linspace(min_val, max_val,200)
        bins = [-1, 0.87949 ,0.96184 ,1]
        #bins = [-0.5,0.5,1.5,2.5,3.5,4.5]
        #plot_range = (min(bins)-0.001, max(bins))
        #bins[0] = np.percentile(values, 0)
        #bins[len(bins)-1] = np.percentile(values, 100)
        
        plot_range = [0.86,1]
        plotVariables(df, variable,bins,plot_range)

def plotVariables(df, variable,bins,plot_range):

#bins = np.linspace(min(df[variable]), max(df[variable]),16)
    
    # Initialise plot stuff
    plt.ion()
    plt.clf()
    #plot_range = (min(df[variable]), max(df[variable]))
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"
    
    df_signal = df.loc[df['Class'] ==1.0]
    df_background = df.loc[df['Class'] ==0.0]

    plot_weights_signal = df_signal['post_fit_weight']
    plot_weights_signal = (1/sum(plot_weights_signal))*plot_weights_signal

    plot_weights_background = df_background['post_fit_weight']
    plot_weights_background = (1/sum(plot_weights_background))*plot_weights_background

    variable_values_signal = df_signal[variable]
    variable_values_background = df_background[variable]

    plot_weights.append(plot_weights_signal.tolist())
    plot_weights.append(plot_weights_background.tolist())

    plot_data.append(variable_values_signal.tolist())
    plot_data.append(variable_values_background.tolist())
                       
    plot_colors.append('#FF0000')
    plot_colors.append('#0066CC')
    
    class_names = ['Background','Signal']

    # Plot.
    plt.hist(variable_values_signal,
        bins=bins,
        weights=plot_weights_signal,
        range=plot_range,
        rwidth=1,
        color='#FF0000',
        edgecolor='red',
        label='Signal',
        stacked=False,
        alpha=0.5,
        hatch=4*'\\')
    plt.hist(variable_values_background,
             bins=bins,
             weights=plot_weights_background,
             range=plot_range,
             rwidth=1,
             color='#0066CC',
             edgecolor='blue',
             label='Background',
             alpha=0.5,
             hatch=4*'//')
    x1, x2, y1, y2 = plt.axis()
    #plt.yscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([0.001,5])
    #axes.set_xlim(plot_range)
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    #plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    
    axes.set_xlim(plot_range)
    #plt.yticks(yi, y,fontweight = 'normal',fontsize = 16)
    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles[::-1])
    plt.ylabel('Normalised Number of Events',fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.1,0.5)
    plt.xlabel(variable,fontsize = 16,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    plt.savefig("final"+str(nJets)+'jets_'+ variable + '.png', bbox_inches='tight')
    plt.show(block=True)

if __name__ == '__main__':
    main()



