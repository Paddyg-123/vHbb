#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from loadEvents import *
from sensitivity import *
from sklearn.ensemble import GradientBoostingClassifier
import time

variables_map = {
    2: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
    }

nJets = 3
variables = variables_map[nJets]

def main():
    current_milli_time = lambda: int(round(time.time() * 1000))
    print "MVA analysis on the " + str(nJets) + " Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = getPreparedDFs(nJets, 'even')
    df_odd = getPreparedDFs(nJets, 'odd')
    
    df_even = set_training_weights(df_even)
    df_odd = set_training_weights(df_odd)
    
    print "Data frames loaded."
    
    print "Training Classifier..."
    
    i = current_milli_time()
    clf_even = GradientBoostingClassifier(learning_rate=0.15,
                                          max_depth=3, min_samples_leaf=0.05,
                                          n_estimators=200,
                                          loss='exponential'
                                          )
    clf_odd =  GradientBoostingClassifier(learning_rate=0.15,
                                          max_depth=3, min_samples_leaf=0.05,
                                          n_estimators=200,
                                          loss='exponential'
                                          )
    
    clf_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
    clf_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
    
    print "Testing Classifier..."
    scores_even = clf_odd.predict_proba(df_even[variables])[:,1]
    scores_odd = clf_even.predict_proba(df_odd[variables])[:,1]
    f = current_milli_time()
    print f-i
    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)
    
    if True == True:
    # TrafoD and score sensitivity.
        sens, error = calc_sensitivity_with_error(df)
        print "Sensitivity for  " + str(nJets) + " Jet Dataset is: {:f}".format(sens) + " +/- {:f}".format(error)

    if True == False:
        y = df['Class'].tolist()
        y_pred = df['decision_value'].tolist()
        #w = df['EventWeight'].tolist()
        
        w = (df['training_weight']*(df['Class']/1521 + (1-df['Class'])/8.9736)).tolist()
        
        sens = calc_sensitivity_tuples(y,y_pred,w)
        print "sens = " + str(sens)

    #Plot BDT.
    print "Plotting histogram..."
    #decision_plot(df, show=True, block=True)
    print "Script finished."

if __name__ == '__main__':
    main()
    
