#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from sensitivity import *
#from histogramPlotATLAS import *
from finalHistogramPlot import *
from xgboost import XGBClassifier

variables_map = {
    '2': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
    }


nJets = '2'
variables = variables_map[nJets]

def main():
    print "MVA analysis on the " + str(nJets) + " Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)

    print "Data frames loaded."

    print "Training Classifier..."
    #set classifier parameters
    if nJets == '2':
        n_estimators = 150
        max_depth = 5
        learning_rate = 0.05
        subsample=0.5

    if nJets == '3':
        n_estimators = 250
        max_depth = 5
        learning_rate = 0.05
        subsample=0.5

    seed = 6

    xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,subsample=subsample,seed=seed)

    xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,subsample=subsample,seed=seed)

    xgb_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
    xgb_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
    
    print "Testing Classifier..."
    scores_even = xgb_odd.predict_proba(df_even[variables])[:,1]
    scores_odd = xgb_even.predict_proba(df_odd[variables])[:,1]
    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)

    # TrafoD and score sensitivity.
    sensitivity, error = calc_sensitivity_with_error(df)
    print "Sensitivity is: {:f}".format(sensitivity) + " +/- {:f}".format(error)

if __name__ == '__main__':
    main()
    
