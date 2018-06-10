#!/bin/python
"""
    A script that prepares the CSV files ready to by put into MVA
    """
# Authors: Patrick Greenway
# License: MIT
import pandas as pd
import numpy as np

scale_factor_map = { #UPDATE THESE
    2: {
        'Zl': 0.99,
        'Zcl': 1.00,
        'Zcc': 1.37,
        'Zbl': 1.37,
        'Zbc': 1.37,
        'Zbb': 1.37,
        'Wl': 0.93,
        'Wcl': 0.93,
        'Wcc': 1.22,
        'Wbl': 1.22,
        'Wbc': 1.22,
        'Wbb': 1.22,
        'stopWt': 0.97,
        'stopt': 0.97,
        'stops': 0.97,
        'ttbar': 0.92,
        'WW': 0.99,
        'ZZ': 0.99,
        'WZ': 0.99,
        'qqZvvH125': 1.0,
        'qqWlvH125': 1.0,
        'qqZllH125': 1.0,
        'ggZllH125': 1.0,
        'ggZvvH125': 1.0
    }, 3: {
        'Zl': 1.0,
        'Zcl': 1.0,
        'Zcc': 1.20,
        'Zbl': 1.20,
        'Zbc': 1.20,
        'Zbb': 1.20,
        'Wl': 0.95,
        'Wcl': 1.02,
        'Wcc': 1.27,
        'Wbl': 1.27,
        'Wbc': 1.27,
        'Wbb': 1.27,
        'stopWt': 0.94,
        'stopt': 0.94,
        'stops': 0.94,
        'ttbar': 0.92,
        'WW': 0.91,
        'ZZ': 0.91,
        'WZ': 0.91,
        'qqZvvH125': 1.0,
        'qqWlvH125': 1.0,
        'qqZllH125': 1.0,
        'ggZllH125': 1.0,
        'ggZvvH125': 1.0,
    }
}

def getPreparedDFs(nJets, kfold):
    # Load in the NTuple CSVs as DataFrames.
    df = pd.read_csv('../CSV_Results_v14/VHbb_data_' + str(nJets) + 'jet_' + kfold + '.csv', index_col=0)
    
    #Set post fit weights
    sampleList = df['sample'].tolist()
    eventWeightList = df['EventWeight'].tolist()
    postFitWeights = []
    
    for x in xrange(0,len(sampleList)):
        postFitWeights.append(eventWeightList[x]*scale_factor_map[nJets][sampleList[x]])
    
    df['post_fit_weight'] = postFitWeights

    return df

def set_training_weights(df):
    """Takes a list of events and sets their renormalised training weights."""
    
    sig_weight_SUM = sum(df['Class']*df['EventWeight'])
    back_weight_SUM = sum((1-df['Class'])*df['EventWeight'])
    
    sig_FREQ = sum(df['Class'])
    back_FREQ = len(df['Class'])-sig_FREQ
    
    sig_scale = sig_FREQ / sig_weight_SUM
    back_scale = back_FREQ / back_weight_SUM
    
    df['training_weight'] = df['EventWeight'] * (df['Class']*sig_scale + (1-df['Class'])*back_scale)
    
    return df

def get_training_weights(df):
    """Takes a list of events and sets their renormalised training weights."""
    
    sig_weight_SUM = sum(df['Class']*df['EventWeight'])
    back_weight_SUM = sum((1-df['Class'])*df['EventWeight'])
    
    sig_FREQ = sum(df['Class'])
    back_FREQ = len(df['Class'])-sig_FREQ
    
    sig_scale = sig_FREQ / sig_weight_SUM
    back_scale = back_FREQ / back_weight_SUM
    
    return df['EventWeight'] * (df['Class']*sig_scale + (1-df['Class'])*back_scale)

