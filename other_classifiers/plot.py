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

variables_map = {
    2: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
    }

nJets = 2
variables = variables_map[nJets]

def main():
    #Get prepared data frames ready to be inserted into classifier
    df = pd.read_csv('new_bdt_'+ str(nJets)+'.csv', index_col=0)
    final_decision_plot(df, show=True, block=True, trafoD_bins = True, bin_number = 15)
    print "Script finished."

if __name__ == '__main__':
    main()
    
