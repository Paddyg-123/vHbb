import numpy as np
import pandas as pd
#from histogramPlotATLAS import *
from sensitivity import *
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import scale
from keras.utils import plot_model
from keras.callbacks import History


variables_map = {
    2: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    3: ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']
}

nJets = 2
variables = variables_map[nJets]

def main():
    
    # Set parameters
    lam = 400
    np.random.RandomState(21)
    train = 'odd'
    test  = 'even'
    
    # Prepare data
    df_train = pd.read_csv('ADV_2jet_batch_'+train+'.csv', index_col=0)
    df_test = pd.read_csv('ADV_2jet_batch_'+test+'.csv', index_col=0)

    #Convert mass bin number to categorical
    z_train = to_categorical(df_train['mBB_category'], num_classes=10)

    # Set network architectures
    inputs = Input(shape=(df_train[variables].shape[1],))
    Dx = Dense(40, activation="linear")(inputs)
    Dx = Dense(40, activation="tanh")(Dx)
    Dx = Dense(40, activation="tanh")(Dx)
    Dx = Dense(1, activation="sigmoid", name='classifier_output')(Dx)
    D = Model(input=[inputs], output=[Dx])
    D.name = 'D'

    Rx = D(inputs)
    Rx = Dense(30, activation="tanh")(Rx)
    Rx = Dense(10, activation="softmax", name='adversary_output')(Rx)
    R = Model(input=[inputs], output=[Rx])
    R.name = 'R'

    #Build and compile models
    """ Build D (Classifier)"""
    opt_D = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    D.trainable = True; R.trainable = False
    D.compile(loss='binary_crossentropy', optimizer=opt_D, metrics=['binary_accuracy'])
    D.fit(df_train[variables], df_train['Class'], sample_weight=df_train['training_weight'], epochs=30, batch_size=32)
    
    test_classifier(D,df_test,lam,'start',test)

    """ Build DRf (Adversary)"""
    DfR = Model(input=[inputs], output=[R(inputs)])
    # Pretraining of R
    opt_DfR = SGD(lr=1, momentum=0.5, decay=0.00001)
    D.trainable = False; R.trainable = True
    DfR.compile(loss='categorical_crossentropy', optimizer=opt_DfR, metrics=['accuracy'])
    DfR.fit(df_train[variables], z_train, sample_weight=df_train['adversary_weights'],batch_size=128, epochs=30)

    """ Build DRf (Model with combined loss function) """
    opt_DRf = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    DRf = Model(input=[inputs], output=[D(inputs), R(inputs)])
    D.trainable = True; R.trainable = False
    DRf.compile(loss={'D':'binary_crossentropy','R':'categorical_crossentropy'}, optimizer=opt_DRf, metrics=['accuracy'], loss_weights={'D': 1.0,'R': -lam})
    
    #Set to 100 epochs
    max = len(df_train)
    batch_size = 100
    
    #Adversarial training
    for i in range(1,max):
        
        if (i%1000 == 0):
            print "Iteration Number: " + str(i)
        
        indices = np.random.permutation(len(df_train))[:batch_size]
        DRf.train_on_batch(df_train[variables].iloc[indices], [df_train['Class'].iloc[indices], z_train[indices]], sample_weight=[df_train['training_weight'].iloc[indices],df_train['adversary_weights'].iloc[indices]])
        DfR.train_on_batch(df_train[variables].iloc[indices], z_train[indices], df_train['adversary_weights'].iloc[indices])
        
        if (i%2500 == 0):
            test_classifier(D,df_test,lam,str(i),test)

    test_classifier(D,df_test,lam,"end",test)
    D.save_weights(str(nJets)+"_"+str(lam)+"_"+train+".h5")






def test_classifier(D,df,lam,stage,test):    #tests classifier model and saves results
    scores = D.predict(df[variables])[:,0]
    df['decision_value'] = ((scores-0.5)*2)
    sensitivity, error = calc_sensitivity_with_error(df)
    print 'Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error)
    df['sensitivity'] = sensitivity
    df.to_csv(path_or_buf='adv_results/'+test+'_batch_'+stage+'_'+str(lam)+'.csv')



if __name__ == '__main__':
    main()

