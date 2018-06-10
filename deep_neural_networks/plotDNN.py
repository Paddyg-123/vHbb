import sys
sys.path.append("/Users/patrickgreenway/ENV/lib/python2.7/site-packages")

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sensitivity import *
from finalHistogramPlotDNN import *


df = pd.read_csv('DNN_2jet_.csv', index_col=0)
df = df.reset_index(drop=True)
#sensitivity, error = calc_sensitivity_with_error(df)
#print "Sensitivity forJet Dataset is: {:f}".format(sensitivity) + " +/- {:f}".format(error)

final_decision_plot(df, show=True, block=True, trafoD_bins = True, bin_number = 20)


