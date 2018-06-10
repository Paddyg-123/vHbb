import numpy as np
import pandas as pd
import matplotlib as plt
import parula as par
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from histogramPlotATLAS import *


lam = 2
parula_cmap = par.parula_map
fig, axs = plt.subplots(figsize=(12, 4), ncols=2)
df1 = pd.read_csv('adv_results/even_batch_0_140.csv', index_col=0)
#df1 = pd.read_csv('adv_results/epochs_pre_'+str(lam)+'.csv', index_col=0)
df1 = df1.loc[df1['Class'] == 0]
y1 = ((df1['decision_value']/2)+0.5).tolist()
x1 = (df1['mBB_raw']/(df1['mBB_raw']+125000)).tolist()
w1 = df1['post_fit_weight'].tolist()

ax = axs[0]
hb = ax.hist2d(x1, y1, bins=(20,20),weights=w1,cmap = parula_cmap)
plt.colorbar(hb[3],ax=ax)
sens1 = round(1.41*(df1['sensitivity'].tolist()[0]),3)
ax.set_title('Classifier; Sensitivity = ' + str(sens1))
ax.set_ylabel("NN output")
ax.set_xlabel("mBB / (mBB + 125 GeV)")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])


df3 = pd.read_csv('adv_results/even_batch_35000_140.csv', index_col=0)
#df2 = pd.read_csv('adv_results/epochs_post_'+str(lam)+'.csv', index_col=0)
df2 = df3.loc[df3['Class'] == 0]
y2 = ((df2['decision_value']/2)+0.5).tolist()
x2 = (df2['mBB_raw']/(df2['mBB_raw']+125000)).tolist()
w2 = df2['post_fit_weight'].tolist()

ax = axs[1]
hb = ax.hist2d(x2, y2, bins=(20,20),weights=w2,cmap = parula_cmap)
plt.colorbar(hb[3],ax=ax)
sens2 = round(1.41*(df2['sensitivity'].tolist()[0]),3)
ax.set_title('Classifier + Adversary; Sensitivity = ' + str(sens2))
ax.set_ylabel("NN output")
ax.set_xlabel("mBB / (mBB + 125 GeV)")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.show()
decision_plot(df3, show=True, block=True, bin_number = 20)

