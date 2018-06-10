import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sb

from sensitivity import *
from datetime import datetime

process_color_map =   {'WW': '#333333',
                        'ggZllH125': '#FF0000',
                        'ggZvvH125': '#FF0000',
                        'qqWlvH125': '#FF0000',
                        'qqZllH125': '#FF0000',
                        'qqZvvH125': '#FF0000',
                        'WZ': '#CCCCCC',
                        'Wbb': '#006600',
                        'Wbc': '#007700',
                        'Wbl': '#009900',
                        'Wcc': '#00CC00',
                        'Wcl': '#66CC66',
                        'Wl': '#99FF99',
                        'ZZ': '#999999',
                        'Zbb': '#0066CC',
                        'Zbc': '#0066CC',
                        'Zbl': '#3399FF',
                        'Zcc': '#6699FF',
                        'Zcl': '#6699CC',
                        'Zl': '#99CCFF',
                        'stopWt': '#FFFF66',
                        'stops': '#CC9900',
                        'stopt': '#CC9900',
                        'ttbar': '#FFCC00'}

class_names = ['WZ', 'ZZ', 'WW', 'stopWt', 'stops', 'stopt', 'ttbar',
             'Zl', 'Zcl', 'Zbl', 'Zcc', 'Zbc', 'Zbb', 'Wl', 'Wcl',
             'Wbl', 'Wcc', 'Wbc', 'Wbb', 'qqWlvH125']

class_names_legend = ['VH -> Vbb','Diboson','ttbar','Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)','Z+cl','Z+ll']

class_names_map = {'VH -> Vbb':['ggZllH125','ggZvvH125','qqWlvH125', 'qqZllH125', 'qqZvvH125'],
    'Diboson':['WW','ZZ','WZ'],
    'ttbar':'ttbar',
    'Single top':'stop',
    'W+(bb,bc,cc,bl)':['Wbb','Wbc','Wcc','Wbl'],
    'W+cl':'Wcl',
    'W+ll':'Wl',
    'Z+(bb,bc,cc,bl)':['Zbb','Zbc','Zcc','Zbl'],
    'Z+cl':'Zcl',
    'Z+ll':'Zl'}

colour_map = {'VH -> Vbb':'#FF0000',
    'Diboson':'#999999',
    'ttbar':'#FFCC00',
    'Single top':'#CC9900',
    'W+(bb,bc,cc,bl)':'#006600',
    'W+cl':'#66CC66',
    'W+ll':'#99FF99',
    'Z+(bb,bc,cc,bl)':'#0066CC',
    'Z+cl':'#6699CC',
    'Z+ll':'#99CCFF'}
    
process_general_map = {
        'ggZllH125': 'VH',
        'ggZvvH125': 'VH',
        'qqZvvH125': 'VH',
        'qqWlvH125': 'VH',
        'qqZllH125': 'VH',
        'Wbb': 'V+jets',
        'Wbc': 'V+jets',
        'Wcc': 'V+jets',
        'Wbl': 'V+jets',
        'Wcl': 'V+jets',
        'Wl': 'V+jets',
        'Zbb': 'V+jets',
        'Zbc': 'V+jets',
        'Zcc': 'V+jets',
        'Zbl': 'V+jets',
        'Zcl': 'V+jets',
        'Zl': 'V+jets',
        'ttbar': 'ttbar',
        'stopt': 'stop',
        'stops': 'stop',
        'stopWt': 'stop',
        'WW': 'diboson',
        'ZZ': 'diboson',
        'WZ': 'diboson'}

def decision_plot(df, show=False, block=False, trafoD_bins = False, bin_number = 15):
    """Plots histogram decision score output of classifier"""
    
    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df)
    else:
         bins = np.linspace(-1,1,bin_number+1)
    
    # Initialise plot stuff
    plt.ion()
    plt.clf()
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    
    decision_value_list = df['decision_value'].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()
    
    # Get list of hists.
    for c in class_names:
        class_decision_vals = []
        plot_weight_vals = []
        
        for x in xrange(0,len(decision_value_list)):
            if sample_list[x] == c:
                class_decision_vals.append(decision_value_list[x])
                plot_weight_vals.append(post_fit_weight_list[x])
    
        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(process_color_map[c])

    # Plot.
    plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             range=plot_range,
             rwidth=1,
             color=plot_colors,
             label=class_names,
             stacked=True,
             edgecolor='None')
    x1, x2, y1, y2 = plt.axis()
    #plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    #axes.set_ylim([10,1000000])
    plt.legend(loc='upper right', ncol=2, prop={'size': 12})
    plt.ylabel('Events')
    plt.xlabel('Score')
    plt.title('Decision Scores')

    plt.show(block=block)





