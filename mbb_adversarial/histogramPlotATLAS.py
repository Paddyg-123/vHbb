import sys
sys.path.append("/Users/patrickgreenway/ENV/lib/python2.7/site-packages")

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sensitivity import *

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


def decision_plot(df, show=False, block=False, trafoD_bins = False, bin_number = 15):
    """Plots histogram decision score output of classifier"""
    
    nJets = df['nJ'].tolist()[1]
    
    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df)
    else:
         bins = np.linspace(-1,1,bin_number+1)
    
    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"
#plt.rcParams["font.family"] = "sans-serif"
    #plt.rcParams['font.sans-serif'] = "Helvetica"

#hfont = {'fontname':'Helvetica'}



    decision_value_list = df['decision_value'].tolist()
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
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,100000])
    axes.set_xlim([-1,1])
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 16)
    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles[::-1])
    plt.ylabel("Events",fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(r"BDT$_{VH}$ output",fontsize = 16,fontweight='normal')
    #plt.xlabel(r"BDT output",fontsize = 16,fontweight='normal')
    #print "changed line"
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    
    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    plt.show(block=block)

def ttbar_plot(df, show=False, block=False, trafoD_bins = False, bin_number = 15):
    """Plots histogram decision score output of classifier"""
    
    nJets = df['nJ'].tolist()[1]
    
    if trafoD_bins == True:
        bins, arg2, arg3 = trafoD_with_error(df)
    else:
        bins = np.linspace(-1,1,bin_number+1)

    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"
    #plt.rcParams["font.family"] = "sans-serif"
    #plt.rcParams['font.sans-serif'] = "Helvetica"

    #hfont = {'fontname':'Helvetica'}



    decision_value_list = df['decision_value'].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()

    class_names_grouped = ['VH -> Vbb','ttbar']
    legend_names = [r'VH $\rightarrow$ Vbb',r"t$\bar t$"]
    
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
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,100000])
    axes.set_xlim([-1,1])
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 16)
    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles[::-1])
    plt.ylabel("Events",fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(r"BDT$_{VH}$ output",fontsize = 16,fontweight='normal')
    #plt.xlabel(r"BDT output",fontsize = 16,fontweight='normal')
    #print "changed line"
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)
    
    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)
    
    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    
    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    
    plt.show(block=block)

def variable_plot(df, variable ,show=False, block=False, trafoD_bins = False, bin_number = 15):
    """Plots histogram decision score output of classifier"""
    
    nJets = df['nJ'].tolist()[1]

    # Initialise plot stuff
    plt.ion()
    plt.close("all")
    plt.figure(figsize=(8.5,7))
    plot_range = (-1, 1)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"

    variable_value_list = df[variable].tolist()
    post_fit_weight_list = df['post_fit_weight'].tolist()
    sample_list = df['sample'].tolist()
    
    # Get list of hists.
    for t in class_names_grouped[::-1]:
        class_names = class_names_map[t]
        class_variable_vals = []
        plot_weight_vals = []
        for c in class_names:
            for x in xrange(0,len(variable_value_list)):
                if sample_list[x] == c:
                    class_variable_vals.append(variable_value_list[x])
                    plot_weight_vals.append(post_fit_weight_list[x])
        
        plot_data.append(class_variable_vals)
        plot_weights.append(plot_weight_vals)
        plot_colors.append(colour_map[t])
    
    maxVal = max(variable_value_list)
    #maxVal = 250000
    minVal = min(variable_value_list)
    bins = np.linspace(minVal,maxVal,21)

    plot_range = (minVal,maxVal)

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
    plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([5,100000])
    axes.set_xlim(plot_range)
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    #plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    plt.yticks(yi, y,fontweight = 'normal',fontsize = 16)
    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
               handles=handles[::-1])
    plt.ylabel("Events",fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.07,0.93)
    plt.xlabel(variable,fontsize = 16,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)

    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)

    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)

    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    
    plt.show(block=block)



