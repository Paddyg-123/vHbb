#!/bin/python
"""
    executable script that plots a 2d histogram of mBB against NN output, before
    and after adversarial training
    """
# Authors: Patrick Greenway

import numpy as np
import pandas as pd
import matplotlib as plt
import parula as par
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from histogramPlotATLAS import *
from sensitivity import *

from scipy.stats import pearsonr

lam = 400
parula_cmap = par.parula_map

initial = 'start'
final = 'end'

df1_even = pd.read_csv('adv_results/even_batch_'+initial+'_'+str(lam)+'.csv', index_col=0)
df1_odd = pd.read_csv('adv_results/odd_batch_'+initial+'_'+str(lam)+'.csv', index_col=0)
df1 = df1_odd.append(df1_even)
df1 = df1.reset_index(drop=True)

xb = df1['mBB_raw'].tolist()
yb = df1['decision_value'].tolist()

print pearsonr(xb,yb)

df2_even = pd.read_csv('adv_results/even_batch_'+final+'_'+str(lam)+'.csv', index_col=0)
df2_odd = pd.read_csv('adv_results/odd_batch_'+final+'_'+str(lam)+'.csv', index_col=0)
df2 = df2_odd.append(df2_even)
df2 = df2.reset_index(drop=True)

xa = df2['mBB_raw'].tolist()
ya = df2['decision_value'].tolist()

print pearsonr(xa,ya)


if True == False:
    sensitivity1, error1 = calc_sensitivity_with_error(df1)
    print sensitivity1
    print error1
    sensitivity2, error2 = calc_sensitivity_with_error(df2)
    print sensitivity2
    print error2
else:
    sensitivity1 = 2.53952263647
    error1 = 0.0748244991203
    sensitivity2 = 1.30825429698
    error2 = 0.020386005895

df1 = df1.loc[df1['Class'] == 0]
y1 = ((df1['decision_value']/2)+0.5).tolist()
x1 = (df1['mBB_raw']/(df1['mBB_raw']+125000)).tolist()
w1 = df1['post_fit_weight'].tolist()

df2 = df2.loc[df2['Class'] == 0]
#df2 = df2.loc[df2['decision_value'] > -1]
y2 = ((df2['decision_value']/2)+0.5).tolist()
x2 = (df2['mBB_raw']/(df2['mBB_raw']+125000)).tolist()
w2 = df2['post_fit_weight'].tolist()


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["mathtext.default"] = "regular"

fig, axs = plt.subplots(figsize=(12, 4), ncols=2)
ax = axs[0]
hb = ax.hist2d(x1, y1, bins=(20,20),weights=w1,cmap = parula_cmap)
plt.colorbar(hb[3],ax=ax,label = 'Events')

#ax.set_title('Classifier; Sensitivity = ' + str(round(sensitivity1,3)))
#ax.set_title("Classifier")
ax.set_ylabel("NN output")
ax.set_xlabel("$m_{bb} / (m_{bb} + 125 GeV)$")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax = axs[1]
hb = ax.hist2d(x2, y2, bins=(20,20),weights=w2,cmap = parula_cmap)
plt.colorbar(hb[3],ax=ax,label = 'Events')
#ax.set_title('Classifier + Adversary; Sensitivity = ' + str(round(sensitivity2,3)))
#ax.set_title("Classifier + Adversary")
ax.set_ylabel("NN output")
ax.set_xlabel("$m_{bb} / (m_{b} + 125 GeV)$")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.show()




