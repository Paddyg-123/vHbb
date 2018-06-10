import numpy as np
import pandas as pd
import matplotlib as plt

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

lam = 0

fig, axs = plt.subplots(figsize=(12, 4), ncols=2)

df1 = pd.read_csv('with_mBB.csv', index_col=0)
df1 = df1.loc[df1['Class'] == 0]
y1 = df1['decision_value'].tolist()
x1 = (df1['mBB']/(df1['mBB']+125000)).tolist()

ax = axs[0]
hb = ax.hist2d(x1, y1, bins=(20,20))
plt.colorbar(hb[3],ax=ax)
ax.set_title('Classifier')
ax.set_ylabel("NN output")
ax.set_xlabel("mBB / (mBB + 125 GeV)")
ax.set_xlim([0, 1])
ax.set_ylim([-1, 1])

df2 = pd.read_csv('without_mBB.csv', index_col=0)
df2 = df2.loc[df2['Class'] == 0]
y2 = df2['decision_value'].tolist()
x2 = (df2['mBB']/(df2['mBB']+125000)).tolist()

ax = axs[1]
hb = ax.hist2d(x2, y2, bins=(20,20))
plt.colorbar(hb[3],ax=ax)
ax.set_title('Classifier + adversary')
ax.set_ylabel("NN output")
ax.set_xlabel("mBB / (mBB + 125 GeV)")
ax.set_xlim([0, 1])
ax.set_ylim([-1, 1])

plt.show()

