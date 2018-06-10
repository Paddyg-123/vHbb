import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
}

nJets = 2
variables = variables_map[nJets]
variables = ['MV1cB1']

def main():
    #df1 = pd.read_csv('allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    #df2 = pd.read_csv('allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    
    df1 = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    df2 = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    
    df = df1.append(df2)
    df = df.reset_index(drop=True)

    df_1 = df.loc[df['sample']==]
    df_2 = df
    ratio = sum(df_1['EventWeight'])/sum(df_2['EventWeight'])
    print ratio


if __name__ == '__main__':
    main()



