import sys
sys.path.append("/Users/patrickgreenway/ENV/lib/python2.7/site-packages")

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sensitivity_mbb import *

class_names_grouped = ['VH -> Vbb','Diboson','ttbar','Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)',
                       'Z+cl','Z+ll'
                       ]

class_names_map = {'VH -> Vbb':['ggZllH125','ggZvvH125','qqWlvH125', 'qqZllH125', 'qqZvvH125'],
    'Diboson':['WW','ZZ','WZ'],
    'ttbar':['ttbar'],
    'Single top':['stopWt','stops','stopt'],
    'W+(bb,bc,cc,bl)':['Wbb','Wbc','Wcc','Wbl'],
    'W+cl':['Wcl'],
    'W+ll':['Wl'],
    'Z+(bb,bc,cc,bl)':['Zbb','Zbc','Zcc','Zbl'],
    'Z+cl':['Zcl'],
    'Z+ll':['Zl']
}

colour_map = {'VH -> Vbb':'#FF0000',
    'Diboson':'#999999',
    'ttbar':'#FFCC00',
    'Single top':'#CC9900',
    'W+(bb,bc,cc,bl)':'#006600',
    'W+cl':'#66CC66',
    'W+ll':'#99FF99',
    'Z+(bb,bc,cc,bl)':'#0066CC',
    'Z+cl':'#6699CC',
    'Z+ll':'#99CCFF'
}

legend_names = [r'VH $\rightarrow$ Vbb','Diboson',r"t$\bar t$",'Single top', 'W+(bb,bc,cc,bl)','W+cl','W+ll','Z+(bb,bc,cc,bl)',
                'Z+cl','Z+ll'
                ]


def decision_plot_mBB(df, show=False, block=False, trafoD_bins = False, bin_number = 15):
    """Plots histogram decision score output of classifier"""
    
    nJets = df['nJ'].tolist()[1]
    if trafoD_bins == True:
      bins, arg2, arg3 = trafoD_with_error_mbb(df)
    else:
      bins = np.linspace(35,600,bin_number+1)

    for i in range(len(bins)):
        bins[i] = bins[i]/1000

    mbb_vals = df['mBB_raw'].tolist()

    print mbb_vals
    for i in range(len(df)):
        if mbb_vals[i]>=500000:
            mbb_vals[i] = 499500

    df['mBB_raw'] = mbb_vals

    bins = np.linspace(0,500,26)

    #bins = [0, 62806.2806280628, 84008.40084008401, 98009.800980098, 108010.80108010801, 113611.3611361136, 117611.76117611761, 120812.0812081208, 123612.3612361236, 126412.6412641264, 129212.9212921292, 132013.201320132, 136013.601360136, 143614.3614361436, 178417.8417841784, 282828.2828282828, 376437.6437643764, 446044.60446044605, 547654.7654765476, 698069.806980698, 4005000]
    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    plt.figure(figsize=(8.5,7))
    plot_range = (0, 500)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"

    decision_value_list = (df['mBB_raw']/1000).tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_decision_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in xrange(0,len(decision_value_list)):
                if sample_list[x] == c:
                    class_decision_vals.append(decision_value_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])
        
        plot_data.append(class_decision_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])

    # Plot.
    plt.hist(plot_data,
             bins=bins,
             weights=plot_weights,
             range=plot_range,
             rwidth=1,
             color=plot_colors,
             label=legend_names[::-1],
             stacked=True,
             edgecolor='None')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([2,320])
    axes.set_xlim(plot_range)

    df_sig = df.loc[df['Class']==1]
    plt.hist(df_sig['mBB_raw']/1000,
             bins=bins,
             weights=(df_sig['post_fit_weight']*5).tolist(),
             range=plot_range,
             rwidth=1,
             histtype = 'step',
             linewidth=2,
             color='#FF0000',
             label=r'VH $\rightarrow$ Vbb x 5',
             edgecolor='#FF0000')
    
    axes.yaxis.set_ticks_position('both')
#axes.yaxis.set_tick_params(which='minor', direction='in',length = 4, width = 2)
#axes.yaxis.set_tick_params(which='major', direction='in',length = 8, width = 2)
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
#axes.xaxis.set_tick_params(which='minor', direction='in',length = 4, width = 2)
#axes.xaxis.set_tick_params(which='major', direction='in',length = 8, width = 2)
    axes.xaxis.set_tick_params(which='both',direction = 'in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    axes.yaxis.set_minor_locator(AutoMinorLocator(4))

    handles, labels = axes.get_legend_handles_labels()
    #Weird hack thing to get legend entries in correct order
    handles = handles[::-1]
    handles = handles+handles
    handles = handles[1:12]
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles)
    
    plt.ylabel("Events",fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel("$m_{bb}$ [GeV]",fontsize = 16,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)

    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)
    
    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)
    
    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    
    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)


    plt.show(block=block)

