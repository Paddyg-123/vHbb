#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from control_purity import *
from histogramPlotATLAS import *
from xgboost import XGBClassifier

variables_map = {
    '2': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
    }

#nJets = raw_input("nJets = ")
nJets = '2'
variables = variables_map[nJets]

def main():
    print "MVA analysis on the " + str(nJets) + " Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    print "Data frames loaded."
   
    if True == False:
        df_even['Class'] = ((df_even['sample'] == 'Wbb')|(df_even['sample'] == 'Wbc')|(df_even['sample'] == 'Wcc')|(df_even['sample'] == 'Wbl'))*1
        df_odd['Class'] = ((df_odd['sample'] == 'Wbb')|(df_odd['sample'] == 'Wbc')|(df_odd['sample'] == 'Wcc')|(df_odd['sample'] == 'Wbl'))*1

    if True == True:
        df_even['Class'] = (df_even['category'] == 'stop')*1
        df_odd['Class'] = (df_odd['category'] == 'stop')*1

    if True == False:
        df_even['Class'] = (df_even['category'] == 'ttbar')*1
        df_odd['Class'] = (df_odd['category'] == 'ttbar')*1

    if True == False:
        df_even['Class'] = (df_even['category'] == 'diboson')*1
        df_odd['Class'] = (df_odd['category'] == 'diboson')*1
                            

    df_even = df_even.loc[df_even['Mtop'] >= 225000]
    df_odd = df_odd.loc[df_odd['Mtop'] >= 225000]

    df_even = df_even.loc[df_even['mBB'] >= 150000]
    df_odd = df_odd.loc[df_odd['mBB'] >= 150000]

#df_even = df_even.loc[df_even['mBB'] <= 75000]
#df_odd = df_odd.loc[df_odd['mBB'] <= 75000]

#df_even = df_even.loc[df_even['category'] == 'stop']
#df_odd = df_odd.loc[df_odd['category'] == 'stop']
    
    #df_even = df_even.loc[df_even['Mtop'] >=225000]
    #df_odd = df_odd.loc[df_odd['Mtop'] >=225000]
    
    #df_even = df_even.loc[df_even['pTV'] >=225000]
    #df_odd = df_odd.loc[df_odd['pTV'] >=225000]
    
    print "Training Classifier..."
    #set classifier parameters
    if nJets == '2':
        n_estimators = 140
        max_depth = 6
        learning_rate = 0.05
        subsample = 0.90
        gamma = 19
        
    if nJets == '3':
        n_estimators = 240
        max_depth = 4
        learning_rate = 0.1
        subsample = 1
        gamma = 0

    xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, gamma=gamma)

    xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, gamma=gamma)

    xgb_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
    xgb_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
    
    print "Testing Classifier..."
    scores_even = xgb_odd.predict_proba(df_even[variables])[:,1]
    scores_odd = xgb_even.predict_proba(df_odd[variables])[:,1]

    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)

    # TrafoD and score sensitivity.
    purity = calc_bin_purity(df,numberOfBins = 3)
    print purity

    #Plot BDT.
    print "Plotting histogram..."
#decision_plot(df, show=True, block=True, bin_number = 20,trafoD_bins = True)
    from skTMVA import convert_bdt_sklearn_tmva
    convert_bdt_sklearn_tmva(xgb_even, [('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F'),('var1', 'F'), ('var2', 'F')], 'stop_control_even.xml')
    print "Script finished."

if __name__ == '__main__':
    main()
    
