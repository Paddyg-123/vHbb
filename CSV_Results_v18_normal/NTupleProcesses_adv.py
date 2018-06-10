import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages")
sys.path.append("/Library/Python/2.7/site-packages")

import numpy as np
import pandas as pd

from root_numpy import root2array

branch_names = ["sample", "EventWeight", "EventNumber", 'RunNumber', "ChannelNumber","DSID" ,"isOdd", "weight", "nJ", "nTags", "nSigJet", "nForwardJet", 'nTrackJets', 'nTrackJetsOR', "mBB", "dRBB", "dPhiBB", "dEtaBB", "sumPt", "pTB1", "pTB2","J1_Flav","J2_Flav" ,"pTBB", "pTBBoverMET", "etaB1", "etaB2", "MET", 'TruthMET', 'TruthMETX', 'TruthMETY', 'jetnumbertruth20', 'jetnumbertruth30', "MPT", "HT", "METHT", "MV1cB1", "MV1cB2", "MV1cJ3", "MV1cB1_cont", "MV1cB2_cont", "MV1cJ3_cont","pTJ3", "etaJ3", "dRB1J3", "dRB2J3", "mBBJ", 'Aplanarity', 'Sphericity', 'Centrality', 'Planarity', 'Var_C', 'Var_D', "dPhiVBB", "dPhiMETMPT", "dPhiMETdijet", "mindPhi", "BDT", "dPhiLBmin", "Mtop", "dYWH", "dEtaWH", "dPhiLMET", "pTL", "etaL", "mTW", "pTV", 'HTXS_Higgs_pt', 'HTXS_Higgs_eta', 'HTXS_Higgs_m', 'HTXS_V_pt', 'HTXS_V_eta', 'HTXS_V_phi', 'HTXS_V_m', 'HTXS_prodMode', 'HTXS_Stage0_Category', 'HTXS_Stage1_Category_pTjet25', 'HTXS_Stage1_Category_pTjet30', 'HTXS_Njets_pTjet25', 'HTXS_Njets_pTjet30']


# Read in NTuples.
# Output S&B as pseudo 2D ndarrays (array of tuple rows).
signal_direct = root2array("Direct_Signal.root",
                           treename="Nominal",
                           branches=branch_names)

background = root2array("background_Normal.root",
                        treename="Nominal",
                        branches=branch_names)

print "NTuple read-in complete."

# Configure as DataFrames.
signal_direct_df = pd.DataFrame(signal_direct, columns=branch_names)
signal_direct_df['Class'] = pd.Series(np.ones(len(signal_direct_df)))

background_df = pd.DataFrame(background, columns=branch_names)
background_df['Class'] = pd.Series(np.zeros(len(background_df)))

# Concatenate S&B dfs.
df = pd.concat([signal_direct_df, background_df])

# Cutflow.
df = df[df['nTags'] == 2]

# Split into 2 jet and 3 jet trainings.
df_2jet = df[df['nJ'] == 2]
df_3jet = df[df['nJ'] == 3]

# Drop unneeded columns for the training.
df_2jet = df_2jet.drop(['RunNumber', 'nTrackJets', 'TruthMET', 'TruthMETX', 'TruthMETY', 'jetnumbertruth20', 'jetnumbertruth30', 'Aplanarity', 'Sphericity', 'Centrality', 'Planarity', 'Var_C', 'Var_D', 'HTXS_Higgs_pt', 'HTXS_Higgs_eta', 'HTXS_Higgs_m', 'HTXS_V_pt', 'HTXS_V_eta', 'HTXS_V_phi', 'HTXS_V_m', 'HTXS_prodMode', 'HTXS_Stage0_Category', 'HTXS_Stage1_Category_pTjet25', 'HTXS_Stage1_Category_pTjet30', 'HTXS_Njets_pTjet25', 'HTXS_Njets_pTjet30', "dEtaBB", "dPhiBB", "weight", "dPhiLMET", "BDT", "pTL", "etaL", "sumPt", "ChannelNumber", "isOdd", "nSigJet", "nForwardJet", "pTBB", "pTBBoverMET", "etaB1", "etaB2", "dEtaBB", "HT", "METHT",'MV1cJ3', "MPT", "etaJ3", "dRB1J3", "dRB2J3", "dPhiMETMPT", "dPhiMETdijet", "mindPhi", "dPhiLMET", 'mBBJ', 'pTJ3',"DSID","J1_Flav","J2_Flav","MV1cB1", "MV1cB2", "MV1cJ3_cont"], axis=1)

df_2jet = df_2jet.drop(['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'dRBB', 'pTB1', 'pTB2','dPhiLBmin', 'Mtop', 'dYWH', 'mTW'], axis=1)

df_3jet = df_3jet.drop(['RunNumber', 'nTrackJets', 'TruthMET', 'TruthMETX', 'TruthMETY', 'jetnumbertruth20', 'jetnumbertruth30', 'Aplanarity', 'Sphericity', 'Centrality', 'Planarity', 'Var_C', 'Var_D', 'HTXS_Higgs_pt', 'HTXS_Higgs_eta', 'HTXS_Higgs_m', 'HTXS_V_pt', 'HTXS_V_eta', 'HTXS_V_phi', 'HTXS_V_m', 'HTXS_prodMode', 'HTXS_Stage0_Category', 'HTXS_Stage1_Category_pTjet25', 'HTXS_Stage1_Category_pTjet30', 'HTXS_Njets_pTjet25', 'HTXS_Njets_pTjet30', "dEtaBB", "dPhiBB", "weight", "dPhiLMET", "BDT", "pTL", "etaL", "sumPt", "ChannelNumber", "isOdd", "nSigJet", "nForwardJet", "pTBB", "pTBBoverMET", "etaB1", "etaB2", "dEtaBB", "HT", "METHT", "MPT", "etaJ3", "dRB1J3", "dRB2J3", "dPhiMETMPT", "dPhiMETdijet", "mindPhi", "dPhiLMET","DSID","J1_Flav","J2_Flav","MV1cB1", "MV1cB2", "MV1cJ3"], axis=1)

df_3jet = df_3jet.drop(['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'dRBB', 'pTB1', 'pTB2','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'mBBJ', 'pTJ3', 'MV1cJ3_cont'], axis=1)

# Split these once again by even/odd event number.
df_2jet_even = df_2jet[df_2jet['EventNumber'] % 2 == 0]
df_2jet_odd = df_2jet[df_2jet['EventNumber'] % 2 == 1]
df_3jet_even = df_3jet[df_3jet['EventNumber'] % 2 == 0]
df_3jet_odd = df_3jet[df_3jet['EventNumber'] % 2 == 1]

if True == True:
    df_2jet_even.to_csv(path_or_buf='ADV_data_2jet_even.csv')
    df_3jet_even.to_csv(path_or_buf='ADV_data_3jet_even.csv')
    df_2jet_odd.to_csv(path_or_buf='ADV_data_2jet_odd.csv')
    df_3jet_odd.to_csv(path_or_buf='ADV_data_3jet_odd.csv')
else:
    df_2jet.to_csv(path_or_buf='VHbb_data_2jet_full.csv')
    df_3jet.to_csv(path_or_buf='VHbb_data_3jet_full.csv')

print "NTuple processed to CSV file."
