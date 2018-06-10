import numpy as np
import pandas as pd
import matplotlib as plt
import parula as par
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from histogramPlotATLAS import *

lam = 400

parula_cmap = par.parula_map
fig, axs = plt.subplots(figsize=(12, 6), ncols=7,nrows=4, sharex=True, sharey=True)

test = 'even'

nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

row = 0
for num in nums:
    iter = (num) * 2500 + 40000
    if num == 0:
        iter = 'start'
    if num == 26:
        iter = 'end'
    
    df = pd.read_csv('adv_results/3jet_'+test+'_batch_'+ str(iter) +'_'+str(lam)+'.csv', index_col=0)

    df = df.loc[df['Class'] == 0]
    y1 = ((df['decision_value']/2)+0.5).tolist()
    x1 = (df['mBB_raw']/(df['mBB_raw']+125000)).tolist()
    w1 = df['post_fit_weight'].tolist()

    if num>(6+7*row):
        row += 1
    col = num%7
    ax = axs[row,col]
    hb = ax.hist2d(x1, y1, bins=(20,20),weights=w1,cmap = parula_cmap,normed=True)
                   
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sens = df['sensitivity'].tolist()[0]
    ax.set_title(round(sens, 3))

plt.suptitle(str(lam))
plt.show()


