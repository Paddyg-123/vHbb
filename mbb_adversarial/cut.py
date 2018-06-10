import numpy as np
import pandas as pd
import matplotlib as plt
#import parula as par
from histogramPlotmBB import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from sensitivity_mbb import *
from histogramvariables import *

#parula_cmap = par.parula_map
lam = 400

df1_even = pd.read_csv('adv_results/even_batch_end_'+str(lam)+'.csv', index_col=0)
df1_odd = pd.read_csv('adv_results/odd_batch_end_'+str(lam)+'.csv', index_col=0)
df = df1_odd.append(df1_even)
total_sig = sum(df['EventWeight']*df['Class'])
total_back = sum(df['EventWeight']*(1-df['Class']))

cuts = [-1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.85,0.9,0.95]

if True == False:
    for cut in cuts:
        df1_even = pd.read_csv('adv_results/even_batch_end_'+str(lam)+'.csv', index_col=0)
        df1_odd = pd.read_csv('adv_results/odd_batch_end_'+str(lam)+'.csv', index_col=0)
        df1 = df1_odd.append(df1_even)

        df = df1.loc[df1['decision_value'] >= cut]
        df = df.reset_index(drop=True)
        
        sig_perc = round(100*sum(df['EventWeight']*df['Class'])/total_sig,2)
        back_perc = round(100*sum(df['EventWeight']*(1-df['Class']))/total_back,2)
        sensitivity, error = calc_sensitivity_with_error_mbb(df)
        print "Cut = "+str(cut)+"; Signal Percentage = " + str(sig_perc) + "; Background Percentage = " + str(back_perc)+"; Sensitivity = " + str(sensitivity)
        print error

cut = 0.3

df = df.loc[df['decision_value'] >= cut]
df = df.reset_index(drop=True)
sensitivity, error = calc_sensitivity_with_error_mbb(df)
print "Sensitivity = " + str(sensitivity)
print error
decision_plot_mBB(df,show=True, block=True, trafoD_bins = True, bin_number = 15)
bins = np.linspace(0, 600000,200)
#plotVariables(df, 'mBB_raw',bins,[0,600000])






