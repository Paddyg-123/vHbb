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
    
    #
    lam = 1
    seed = 0; np.random.seed = seed
    
    lambdas = [0.1,0.2,0.3,0.41,0.46,0.51,0.54,0.59,0.63,0.66,0.71,0.74]
    lambdas = [0.1,0.2,0.25,0.3,0.41,0.46,0.51,0.54,0.59,0.63,0.66,0.71,0.74]
    for lam in lambdas:
        print "lambda = " +str(lam)
        # Prepare data
        df_even = pd.read_csv('ADV_2jet_even_batch.csv', index_col=0)
        df_odd = pd.read_csv('ADV_2jet_odd_batch.csv', index_col=0)

        if True == True:
            #Block sets training weights to 0 if negative or >1*10^-3
            training_weights = df_even['training_weight'].tolist()
            adversary_weights = df_even['adversary_weights'].tolist()
            sum_adv = sum(adversary_weights)
            sum_tra = sum(training_weights)
            for i in range(len(df_even)):
                if training_weights[i] <= 0:
                    training_weights[i] = 0
                if adversary_weights[i] <= 0:
                    adversary_weights[i] = 0
                if training_weights[i] >= 0.001*sum_tra:
                    training_weights[i] = 0
                if adversary_weights[i] >= 0.001*sum_adv:
                    adversary_weights[i] = 0

            df_even['training_weight'] = training_weights
            df_even['adversary_weights'] = adversary_weights  #adversary weights are 1*10^-4 smaller for signal
            df_even = df_even.loc[df_even['training_weight']>0]
            df_even = df_even.reset_index(drop=True)


        #Convert mass bin number to categorical
        z_even = to_categorical(df_even['mBB_category'], num_classes=10)
        for i in range(len(z_even)):
            for j in range(10):
                if z_even[i][j] == 1:   #0.25 value for adjacent bins to the correct
                    if j > 0:
                        z_even[i][j-1] = 0.25
                    if j < 9:
                        z_even[i][j+1] = 0.25

        # Set network architectures
        inputs = Input(shape=(df_even[variables].shape[1],))
        Dx = Dense(3*14, activation="linear")(inputs)
        Dx = Dense(1*14, activation="tanh")(Dx)
        Dx = Dense(3, activation="tanh")(Dx)
        Dx = Dense(1, activation="sigmoid", name='classifier_output')(Dx)
        D = Model(input=[inputs], output=[Dx])
        D.name = 'D'

        Rx = D(inputs)
        Rx = Dense(2*14, activation="tanh")(Rx)
        Rx = Dense(20, activation="tanh")(Rx)
        Rx = Dense(20, activation="tanh")(Rx)
        Rx = Dense(10, activation="softmax", name='adversary_output')(Rx)
        R = Model(input=[inputs], output=[Rx])
        R.name = 'R'

        #Build and compile models
        """ Build D (Classifier)"""
        opt_D = SGD(lr=0.01, momentum=0.7, decay=0.00001)
        D.trainable = True; R.trainable = False
        D.compile(loss='mean_squared_error', optimizer=opt_D, metrics=['binary_accuracy'])
        D.fit(df_even[variables], df_even['Class'], sample_weight=df_even['training_weight'], epochs=15, batch_size=128)

        """ Build DRf (Adversary)"""
        DfR = Model(input=[inputs], output=[R(inputs)])
        # Pretraining of R
        D.trainable = False; R.trainable = True
        DfR.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        DfR.fit(df_even[variables], z_even, sample_weight=df_even['adversary_weights'],batch_size=128, epochs=15)

        test_classifier(D,df_odd,lam,'0')

        """ Build DRf (Model with combined loss function) """
        opt_DRf = SGD(lr=0.01, momentum=0.7, decay=0.00001)
        DRf = Model(input=[inputs], output=[D(inputs), R(inputs)])
        D.trainable = True; R.trainable = False
        DRf.compile(loss={'D':'mean_squared_error','R':'categorical_crossentropy'}, optimizer=opt_DRf, metrics=['accuracy'], loss_weights={'D': 1.0,'R': -lam})
        lam_origin = lam
        #Adversarial training
        max = 8
        for i in range(1,max):
            print "Iteration Number: " + str(i)
            # Fit D - combined loss function
            D.trainable = True; R.trainable = False
            DRf.compile(loss={'D':'mean_squared_error','R':'categorical_crossentropy'}, optimizer=opt_DRf, metrics=['accuracy'], loss_weights={'D': 1.0,'R': -lam})
            DRf.fit(df_even[variables], [df_even['Class'], z_even], sample_weight=[df_even['training_weight'],df_even['adversary_weights']],
                batch_size=128, epochs = 3)
                
            
            test_classifier(D,df_odd,lam_origin,str(i))
            
            if i < max-1:
                # Fit R
                D.trainable = False; R.trainable = True
                DfR.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                DfR.fit(df_even[variables], z_even, sample_weight=df_even['adversary_weights'], epochs=3, batch_size=128)
                lam = lam * 0.9

def test_classifier(D,df_odd,lam,stage):    #tests classifier model and saves results
    scores = D.predict(df_odd[variables])[:,0]
    df_odd['decision_value'] = ((scores-0.5)*2)
    df = df_odd
    sensitivity, error = calc_sensitivity_with_error(df)
    print 'Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error)
    df['sensitivity'] = sensitivity
    df.to_csv(path_or_buf='adv_results/_batch_0.9_decreasing_altered_epochs_'+stage+'_'+str(lam)+'_0.7mom.csv')



if __name__ == '__main__':
    main()

