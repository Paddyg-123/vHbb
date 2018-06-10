#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from sensitivity import *
from finalHistogramPlot import *
#from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time

variables_map = {
    2: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
    }

nJets = 3
variables = variables_map[nJets]

def main():
    current_milli_time = lambda: int(round(time.time() * 1000))
    if nJets == 2:
        n_estimators = 200
    
    if nJets == 3:
        n_estimators = 300



#n_estimators = 200


    print "MVA on the " + str(nJets) + " Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v7/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v7/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)

    i = current_milli_time()
    clf_even = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05),
                      learning_rate=0.15,
                      algorithm="SAMME",
                      n_estimators=n_estimators
                      )
    clf_odd =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05),
                    learning_rate=0.15,
                    algorithm="SAMME",
                    n_estimators=n_estimators
                    )
      
    clf_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
    clf_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
    
    print "Testing Classifier..."
    scores_even = clf_odd.decision_function(df_even[variables])
    scores_odd = clf_even.decision_function(df_odd[variables])
    f = current_milli_time()
    print f-i
    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)
      
    # TrafoD and score sensitivity.
    sens, error = calc_sensitivity_with_error(df)
    print "Sensitivity for  " + str(nJets) + " Jet Dataset is: {:f}".format(sens) + " +/- " + str(error)
      
    #Plot BDT.
    print "Plotting histogram..."
    df.to_csv(path_or_buf='new_new_new_bdt_'+ str(nJets)+'.csv')
    final_decision_plot(df, show=True, block=True, trafoD_bins = True, bin_number = 15)

      #decision_plot(df, show=True, block=True)
    print "Script finished."

if __name__ == '__main__':
    main()
    
