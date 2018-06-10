#!/bin/python
"""
An executable main script to classify the 2 or 3 jet dataset using the XGradient Boosted Classifier
"""
# Authors: Patrick Greenway

from sensitivity import *
from finalHistogramPlot import *

nJets = 3
def main():
    df = pd.read_csv("DNN_"+str(nJets)+'jet_.csv', index_col=0)
    

    sensitivity, error = calc_sensitivity_with_error(df)
    print "Sensitivity for  " + str(nJets) + " Jet Dataset is: {:f}".format(sensitivity) + " +/- {:f}".format(error)
    
    df.to_csv(path_or_buf="DNN_"+str(nJets)+'jet_.csv')
    final_decision_plot(df,  trafoD_bins = True,show=True, block=True)
#final_decision_plot(df,  trafoD_bins = False,show=True, block=True)

if __name__ == '__main__':
    main()
    
