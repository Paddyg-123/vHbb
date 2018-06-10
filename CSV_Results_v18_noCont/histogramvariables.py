import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom

nJets = 3

def main():
    df = pd.read_csv('../CSV_Results_v14/allVariables_'+str(nJets)+'jet_full.csv', index_col=0)
    
    variable = 'nTrackJetsOR'
    bins = [0,1,2,3,4,5]
    plot_range = (min(bins), max(bins))
    plotVariables(df, variable,bins,plot_range)

    variable = 'MV1cB1'
    bins = np.linspace(min(df[variable]), max(df[variable]),16)
    plot_range = (min(bins), max(bins))
    plotVariables(df, variable,bins,plot_range)

    variable = 'MV1cB2'
    bins = np.linspace(min(df[variable]), max(df[variable]),16)
    plot_range = (min(bins), max(bins))
    plotVariables(df, variable,bins,plot_range)

    if nJets == 3:
        variable = 'MV1cJ3'
        bins = np.linspace(min(df[variable]), max(df[variable]),16)
        plot_range = (min(bins), max(bins))
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
    plt.hist(plot_data,
         bins=bins,
        weights=plot_weights,
         range=plot_range,
         rwidth=1,
         color=plot_colors,
         label=class_names,
         stacked=False,
         edgecolor='None')
    x1, x2, y1, y2 = plt.axis()
    #plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([0,1.5])
    #axes.set_xlim(plot_range)
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    #plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
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
    axes.yaxis.set_label_coords(-0.07,0.45)
    plt.xlabel(variable,fontsize = 16,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    plt.show(block=True)

if __name__ == '__main__':
    main()



