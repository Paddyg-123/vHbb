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

process_general_map = {
    'ggZllH125': 'VH',  #added this key
        'ggZvvH125': 'VH',  #and this one
        'qqZvvH125': 'VH',
        'qqWlvH125': 'VH',  #l-lepton channel
        'qqZllH125': 'VH',
        'Wbb': 'V+jets',
        'Wbc': 'V+jets',
        'Wcc': 'V+jets',
        'Wbl': 'V+jets',
        'Wcl': 'V+jets',
        'Wl': 'V+jets',
        'Zbb': 'V+jets',
        'Zbc': 'V+jets',
        'Zcc': 'V+jets',
        'Zbl': 'V+jets',
        'Zcl': 'V+jets',
        'Zl': 'V+jets',
        'ttbar': 'ttbar',
        'ttbar-b': 'ttbar-b',
        'ttbar-c': 'ttbar-c',
        'stopt': 'stop',
        'stops': 'stop',
        'stopWt': 'stop',
        'WW': 'diboson',
        'ZZ': 'diboson',
        'WZ': 'diboson'
}

def main():
    folds = ['2jet_even', '2jet_odd']
    for fold in folds:
        # Load in the NTuple CSVs as DataFrames.
        data = pd.read_csv('ADV_data_'+fold+'.csv', index_col=0)
        data = data.reset_index(drop=True)
        
        #Set post fit weights
        sampleList = data['sample'].tolist()
        eventWeightList = data['EventWeight'].tolist()
        postFitWeights = []
        categoryList = []
        
        for x in xrange(0,len(sampleList)):
            postFitWeights.append(eventWeightList[x]*scale_factor_map[2][sampleList[x]])
            categoryList.append(process_general_map[sampleList[x]])
        
        data['post_fit_weight'] = postFitWeights
        data['category'] = categoryList
        data = set_training_weights_for_all(data)

        data.to_csv(path_or_buf='ADV_allVariables_'+fold+'.csv')


def set_training_weights_for_all(df):
    """Takes a list of events and sets their renormalised training weights."""
    
    sig_weight_SUM = sum(df['Class']*df['EventWeight'])
    back_weight_SUM = sum((1-df['Class'])*df['EventWeight'])
    
    sig_FREQ = sum(df['Class'])
    back_FREQ = len(df['Class'])-sig_FREQ
    
    sig_scale = sig_FREQ / sig_weight_SUM
    back_scale = back_FREQ / back_weight_SUM
    
    df['training_weight'] = df['EventWeight'] * (df['Class']*sig_scale + (1-df['Class'])*back_scale)
    
    return df

if __name__ == '__main__':
    main()

