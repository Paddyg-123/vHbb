import numpy as np
import pandas as pd
import matplotlib as plt
from histogramPlotmBB import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from mBB_plot_advanced import *
from sensitivity_mbb import *

nJets = 2

df = pd.read_csv('DNN_2jet_.csv', index_col=0)
df = df.reset_index(drop=True)
cut = 0.65
df_cut = df.loc[df['decision_value'] >= cut]
df_cut = df_cut.reset_index(drop=True)
sensitivity, error = calc_sensitivity_with_error_mbb(df)
print "Sensitivity = " + str(sensitivity)
#decision_plot_mBB(df,show=True, block=True, trafoD_bins = True, bin_number = 15)

plotmBB(df,df_cut)






