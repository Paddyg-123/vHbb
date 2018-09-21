#!/bin/python
"""
    processes dataframes for use in adversarial strategy
    """
# Authors: Patrick Greenway

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import matplotlib as plt
import parula as par
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
}

nJets = 3
variables = variables_map[nJets]

def main():
    
    # Prepare data
    df = pd.read_csv('../../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    
    df['mBB_raw'] = df['mBB']
    
    df_back = df.loc[df['Class']==0]
    df_back_list = df_back['mBB'].tolist()
    lower_bound = np.percentile(df_back_list, 5)
    upper_bound = np.percentile(df_back_list, 95)
    bins = np.linspace(lower_bound,upper_bound,num=11)
    
    bins[0] = np.percentile(df_back_list, 0)
    bins[1] = np.percentile(df_back_list, 10)
    bins[2] = np.percentile(df_back_list, 20)
    bins[3] = np.percentile(df_back_list, 30)
    bins[4] = np.percentile(df_back_list, 40)
    bins[5] = np.percentile(df_back_list, 50)
    bins[6] = np.percentile(df_back_list, 60)
    bins[7] = np.percentile(df_back_list, 70)
    bins[8] = np.percentile(df_back_list, 80)
    bins[9] = np.percentile(df_back_list, 90)
    bins[10] = np.percentile(df_back_list, 100)
    
    mBB_values = df['mBB'].tolist()
    #lower_bound = np.percentile(mBB_values, 5)
    #upper_bound = np.percentile(mBB_values, 95)
    #bins = np.linspace(lower_bound,upper_bound,num=11)
    
    print bins
    
    category = np.zeros(len(mBB_values))
    for i in range(len(mBB_values)):
        if (mBB_values[i] < bins[1]):
            category[i] = 0
        if (mBB_values[i] >= bins[1] and mBB_values[i] < bins[2]):
            category[i] = 1
        if (mBB_values[i] >= bins[2] and mBB_values[i] < bins[3]):
            category[i] = 2
        if (mBB_values[i] >= bins[3] and mBB_values[i] < bins[4]):
            category[i] = 3
        if (mBB_values[i] >= bins[4] and mBB_values[i] < bins[5]):
            category[i] = 4
        if (mBB_values[i] >= bins[5] and mBB_values[i] < bins[6]):
            category[i] = 5
        if (mBB_values[i] >= bins[6] and mBB_values[i] < bins[7]):
            category[i] = 6
        if (mBB_values[i] >= bins[7] and mBB_values[i] < bins[8]):
            category[i] = 7
        if (mBB_values[i] >= bins[8] and mBB_values[i] < bins[9]):
            category[i] = 8
        if (mBB_values[i] >= bins[9]):
            category[i] = 9

    df['adversary_weights'] = ((1-df['Class'])+(0*df['Class']))*(df['training_weight'])
    df['mBB_category'] = category

    print len(df.loc[df['mBB_category'] == 0])
    print len(df.loc[df['mBB_category'] == 1])
    print len(df.loc[df['mBB_category'] == 2])
    print len(df.loc[df['mBB_category'] == 3])
    print len(df.loc[df['mBB_category'] == 4])
    print len(df.loc[df['mBB_category'] == 5])
    print len(df.loc[df['mBB_category'] == 6])
    print len(df.loc[df['mBB_category'] == 7])
    print len(df.loc[df['mBB_category'] == 8])
    print len(df.loc[df['mBB_category'] == 9])

    df[variables] = scale(df[variables])

#df.to_csv(path_or_buf='ADV_3jet_batch_odd.csv')
    print "not saving df"

if __name__ == '__main__':
    main()

