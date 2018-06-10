import numpy as np
import pandas as pd
import matplotlib as plt
#import parula as par
#from histogramPlotmBB import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from sensitivity_mbb import *

#parula_cmap = par.parula_map

df = pd.read_csv('without_mBB.csv', index_col=0)
total_sig = sum(df['EventWeight']*df['Class'])
total_back = sum(df['EventWeight']*(1-df['Class']))

cuts = [-1,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95]

for cut in cuts:
    df = pd.read_csv('without_mBB.csv', index_col=0)

    df = df.loc[df['decision_value'] >= cut]
    df = df.reset_index(drop=True)
    
    sig_perc = round(100*sum(df['EventWeight']*df['Class'])/total_sig,2)
    back_perc = round(100*sum(df['EventWeight']*(1-df['Class']))/total_back,2)
    print "Signal Percentage = " + str(sig_perc) + "; Background Percentage = " + str(back_perc)

    sensitivity = calc_sensitivity_with_error(df)
    print "Sensitivity = " + str(round(sensitivity,2))


