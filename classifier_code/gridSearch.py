#!/bin/python
"""
grid search of primary XGB variables
"""
# Authors: Patrick Greenway

from sensitivity import *
from xgboost import XGBClassifier

variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
    }

nJets = 2
variables = variables_map[nJets]

def main():
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    #set classifier parameters
  
    learning_rate_map = [0.05,0.1,0.15,0.20]
    max_depth_map = [1,2,3,4,5,6,7,8]
    n_estimators_map = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500]
    
    for x in xrange(0,len(learning_rate_map)):
        learning_rate = learning_rate_map[x]
        print "Current Learning Rate = " + str(learning_rate)
        for y in xrange(0,len(max_depth_map)):
            max_depth = max_depth_map[y]
            print "Current Max Depth = " + str(max_depth)
            for z in xrange(0,len(n_estimators_map)):
                n_estimators = n_estimators_map[z]
                
                xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

                xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            
                xgb_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
                xgb_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
                
                scores_even = xgb_odd.predict_proba(df_even[variables])[:,1]
                scores_odd = xgb_even.predict_proba(df_odd[variables])[:,1]
                df_even['decision_value'] = ((scores_even-0.5)*2)
                df_odd['decision_value'] = ((scores_odd-0.5)*2)
                df = df_even.append(df_odd)

                # TrafoD and score sensitivity.
                sens, error = calc_sensitivity_with_error(df)
                print "Sensitivity for " + str(nJets) + " Jet Dataset at "+str(n_estimators)+" estimators is: {:f}".format(sens) + " +/- {:f}".format(error)
                
                file = open("gridsearch_" + str(nJets) + "_" + str(learning_rate) + ".txt","a")
                file.write("\n" + str(n_estimators) + ',' + str(max_depth) + ',' + str(learning_rate) + ',' + str(sens) + ',' + str(error))
                file.close

    print "Script finished."

if __name__ == '__main__':
    main()
    
