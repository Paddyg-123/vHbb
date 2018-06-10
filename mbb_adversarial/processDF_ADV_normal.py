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

nJets = 2
variables = variables_map[nJets]

def main():
    
    # Prepare data
    df_even = pd.read_csv('../../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    
    df_odd['mBB_raw'] = df_odd['mBB']
    df_even['mBB_raw'] = df_even['mBB']z
    
    mBB_values_even = df_even['mBB'].tolist()
    lower_bound = np.percentile(mBB_values_even, 5)
    upper_bound = np.percentile(mBB_values_even, 95)
    bins = np.linspace(lower_bound,upper_bound,num=11)
    
    print bins
    
    category_even = np.zeros(len(mBB_values_even))
    for i in range(len(mBB_values_even)):
        if (mBB_values_even[i] < bins[1]):
            category_even[i] = 0
        if (mBB_values_even[i] >= bins[1] and mBB_values_even[i] < bins[2]):
            category_even[i] = 1
        if (mBB_values_even[i] >= bins[2] and mBB_values_even[i] < bins[3]):
            category_even[i] = 2
        if (mBB_values_even[i] >= bins[3] and mBB_values_even[i] < bins[4]):
            category_even[i] = 3
        if (mBB_values_even[i] >= bins[4] and mBB_values_even[i] < bins[5]):
            category_even[i] = 4
        if (mBB_values_even[i] >= bins[5] and mBB_values_even[i] < bins[6]):
            category_even[i] = 5
        if (mBB_values_even[i] >= bins[6] and mBB_values_even[i] < bins[7]):
            category_even[i] = 6
        if (mBB_values_even[i] >= bins[7] and mBB_values_even[i] < bins[8]):
            category_even[i] = 7
        if (mBB_values_even[i] >= bins[8] and mBB_values_even[i] < bins[9]):
            category_even[i] = 8
        if (mBB_values_even[i] >= bins[9]):
            category_even[i] = 9

    df_even['adversary_weights'] = ((1-df_even['Class'])+(0.0001*df_even['Class']))*(df_even['training_weight'])
    df_even['mBB_category'] = category_even

    print len(df_even.loc[df_even['mBB_category'] == 0])
    print len(df_even.loc[df_even['mBB_category'] == 1])
    print len(df_even.loc[df_even['mBB_category'] == 2])
    print len(df_even.loc[df_even['mBB_category'] == 3])
    print len(df_even.loc[df_even['mBB_category'] == 4])
    print len(df_even.loc[df_even['mBB_category'] == 5])
    print len(df_even.loc[df_even['mBB_category'] == 6])
    print len(df_even.loc[df_even['mBB_category'] == 7])
    print len(df_even.loc[df_even['mBB_category'] == 8])
    print len(df_even.loc[df_even['mBB_category'] == 9])

    df_even[variables] = scale(df_even[variables])
    df_odd[variables] = scale(df_odd[variables])

    df_even.to_csv(path_or_buf='ADV_2jet_even_normal.csv')
    df_odd.to_csv(path_or_buf='ADV_2jet_odd_normal.csv')

if __name__ == '__main__':
    main()

