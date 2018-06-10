#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from histogramPlotATLAS import *

variables_map = {
    '2': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
}

nJets = 2

def main():
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
   
    df = df_odd.append(df_even)
    
    #df['Class'] = ((df['sample'] == 'Wbb')|(df['sample'] == 'Wbc')|(df['sample'] == 'Wcc')|(df['sample'] == 'Wbl'))*1

#df['Class'] = (df['category'] == 'stop')*1

#df['Class'] = (df['category'] == 'ttbar')*1

#df['Class'] = (df['category'] == 'diboson')*1
    
    df = df.loc[df['Mtop'] >= 225000]

    df = df.loc[df['mBB'] >= 150000]

    #df = df.loc[df['mBB'] <= 75000]

    #df = df.loc[df['category'] == 'stop']
    
    #df = df.loc[df['Mtop'] >=225000]
    
    #df = df.loc[df['pTV'] >=225000]
    
    variable = 'pTV'
    
    variable_plot(df, variable ,show=True, block=True, trafoD_bins = False, bin_number = 15)

if __name__ == '__main__':
    main()
    
