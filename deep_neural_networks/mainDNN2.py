#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from sensitivity import *
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import scale
from keras.models import *
from finalHistogramPlot import *

variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
        }

nJets = 2
variables = variables_map[nJets]

def main():
    #print "MVA analysis on the 2 Jet Dataset"
    #Get prepared data frames ready to be inserted into classifier
    df_even = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_even.csv', index_col=0)
    df_odd = pd.read_csv('../CSV_Results_v18_normal/allVariables_'+str(nJets)+'jet_odd.csv', index_col=0)
    
    #print "Data frames loaded."
    df_even[variables] = scale(df_even[variables])
    df_odd[variables] = scale(df_odd[variables])
    
    inputs_even = Input(shape=(df_even[variables].shape[1],))
    clf_evenx = Dense(12, activation="linear")(inputs_even)
    clf_evenx = Dense(12, activation="tanh")(clf_evenx)
    clf_evenx = Dense(12, activation="tanh")(clf_evenx)
    clf_evenx = Dense(1, activation="sigmoid")(clf_evenx)
    clf_even = Model(input=[inputs_even], output=[clf_evenx])
    clf_even.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    inputs_odd = Input(shape=(df_odd[variables].shape[1],))
    clf_oddx = Dense(12, activation="linear")(inputs_odd)
    clf_oddx = Dense(12, activation="tanh")(clf_oddx)
    clf_oddx = Dense(12, activation="tanh")(clf_oddx)
    clf_oddx = Dense(1, activation="sigmoid")(clf_oddx)
    clf_odd = Model(input=[inputs_odd], output=[clf_oddx])
    clf_odd.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    clf_even.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'], epochs=150, batch_size=32)
    
    clf_odd.fit(df_odd[variables], df_odd['Class'], sample_weight=df_odd['training_weight'], epochs=150, batch_size=32)
    
    #print "Testing Classifier..."
    scores_even = clf_odd.predict(df_even[variables])[:,0]
    scores_odd = clf_even.predict(df_odd[variables])[:,0]
    df_even['decision_value'] = ((scores_even-0.5)*2)
    df_odd['decision_value'] = ((scores_odd-0.5)*2)
    df = df_even.append(df_odd)
    
    # TrafoD and score sensitivity.
    sensitivity, error = calc_sensitivity_with_error(df)
    print "Sensitivity for  " + str(nJets) + " Jet Dataset is: {:f}".format(sensitivity) + " +/- {:f}".format(error)
    
    df.to_csv(path_or_buf="new_DNN_"+str(nJets)+'jet_.csv')

    #Plot BDT.
    #print "Plotting histogram..."
    final_decision_plot(df,  trafoD_bins = True,show=True, block=True)
#   final_decision_plot(df,  trafoD_bins = False,show=True, block=True)
#print "Script finished."

if __name__ == '__main__':
    main()
    
