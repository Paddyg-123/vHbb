#!/bin/python
"""
An executable main script to that isolate a specific region of face space and rich in a specific background type

"""
# Authors: Patrick Greenway

from control_purity import *
from finalHistogramPlot import *
from xgboost import XGBClassifier

variables_map = {
    '2': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
    }
variables_map = {
    '2': ['nTrackJetsOR', 'MV1cB1', 'MV1cB2', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'MV1cB1', 'MV1cB2', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3']
}

#nJets = raw_input("nJets = ")
nJets = '2'
variables = variables_map[nJets]

def main():
    print "MVA analysis on the " + str(nJets) + " Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_noCont/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    print "Data frames loaded."
   
    if True == False:
        df_even['Class'] = ((df_even['sample'] == 'Wbb')|(df_even['sample'] == 'Wbc')|(df_even['sample'] == 'Wcc')|(df_even['sample'] == 'Wbl'))*1
        df_odd['Class'] = ((df_odd['sample'] == 'Wbb')|(df_odd['sample'] == 'Wbc')|(df_odd['sample'] == 'Wcc')|(df_odd['sample'] == 'Wbl'))*1
        df_even = df_even.loc[df_even['Mtop'] >= 225000]
        df_odd = df_odd.loc[df_odd['Mtop'] >= 225000]
        df_even = df_even.loc[df_even['mBB'] <= 75000]
        df_odd = df_odd.loc[df_odd['mBB'] <= 75000]


    if True == False:
        df_even['Class'] = (df_even['category'] == 'stop')*1
        df_odd['Class'] = (df_odd['category'] == 'stop')*1
        #df_even = df_even.loc[df_even['Mtop'] <= 225000]
        #df_odd = df_odd.loc[df_odd['Mtop'] <= 225000]
        df_even = df_even.loc[df_even['mBB'] <= 75000]
        df_odd = df_odd.loc[df_odd['mBB'] <= 75000]

    if True == True:
        df_even['Class'] = (df_even['category'] == 'ttbar')*1
        df_odd['Class'] = (df_odd['category'] == 'ttbar')*1
        df_even = df_even.loc[df_even['Mtop'] <= 225000]
        df_odd = df_odd.loc[df_odd['Mtop'] <= 225000]
        df_even = df_even.loc[df_even['mBB'] <= 75000]
        df_odd = df_odd.loc[df_odd['mBB'] <= 75000]
    


    if True == False:
        df_even['Class'] = (df_even['category'] == 'diboson')*1
        df_odd['Class'] = (df_odd['category'] == 'diboson')*1
    if True == False:
        df_even['Class'] = ((df_even['sample'] == 'Wcl')|(df_even['sample'] == 'Wll'))*1
        df_odd['Class'] = ((df_odd['sample'] == 'Wcl')|(df_odd['sample'] == 'Wll'))*1



    purity_even = sum(df_even['post_fit_weight']*df_even['Class']) /sum(df_odd['post_fit_weight'])
    purity_odd = sum(df_odd['post_fit_weight']*df_odd['Class']) /sum(df_odd['post_fit_weight'])
    
    print "Training Classifier..."
    #set classifier parameters
    if nJets == '2':
        n_estimators = 140 #changed from 160 or 150
        max_depth = 5   #changed from 3 or 4
        learning_rate = 0.05
        
    if nJets == '3':
        n_estimators = 250
        max_depth = 6
        learning_rate = 0.05


    xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    xgb_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'])
    xgb_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'])
    
    print "Testing Classifier..."
    scores_even = xgb_odd.predict_proba(df_even[variables])[:,1]
    scores_odd = xgb_even.predict_proba(df_odd[variables])[:,1]

    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)

    # TrafoD and score sensitivity.
    purity,noE = calc_bin_purity(df,1,20)

    for i in range(len(purity)):
        noE[i] = float(1/float(100)) * purity[i] * noE[i]

#purities.sort(key = lambda t: t[0])

    for i in range(6):
        print str(round_sigfigs(purity[i],3)) + " (" + str(round_sigfigs(noE[i],3)) + ")"

    #Plot BDT.
    print "Plotting histogram..."
#decision_plot(df, show=True, block=True, bin_number = 20,trafoD_bins = True)
    final_decision_plot(df, show=True, block=True, bin_number = 20,trafoD_bins = True)
    print "Script finished."

def round_sigfigs(num, sig_figs):
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))

if __name__ == '__main__':
    main()
    
