#
#       Autoencoder  (semi-supervised learning / anomaly detection)
#
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#


import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import initializers, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model

mpl.use('TkAgg')

# tf.config.experimental.enable_op_determinism()
# keras.utils.set_random_seed(1)


class AutoEncoder():

    def TrainStepOne():
        print("this is an important step, report it in AIBOM")
    
    def __init__(self, input_dim):

        input_layer = Input(shape=(input_dim,))

        layer = Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal())(input_layer)
        layer = Dense(16, activation='relu', kernel_initializer=initializers.RandomNormal())(layer)
        layer = Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal())(layer)

        output_layer = Dense(input_dim, activation='tanh', kernel_initializer=initializers.RandomNormal())(layer)

        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)

    def summary(self, ):
        self.autoencoder.summary()

    def train(self, x, y):

        e = 1
        b = 54
        v = 0.1

        self.autoencoder.compile(optimizer='Adam', loss='huber')
        device = tf.config.list_physical_devices()

        print(len(device))

        history = self.autoencoder.fit(x, y, epochs=e, batch_size=b, validation_split=v, shuffle=True, verbose=2)

        validation_split = v

        # -----------------------------------------------#
        #           instructor-provided code             #
        # -----------------------------------------------#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

        #   Computation of the detection threshold with a percentage
        #       of the training set equal to 'validation_split'

        x_thSet = x[x.shape[0]-(int)(x.shape[0]*validation_split):x.shape[0]-1, :]
        self.threshold = self.computeThreshold(x_thSet)

        df_history = pd.DataFrame(history.history)
        return df_history

    def predict(self, x_evaluation):

        predictions = self.autoencoder.predict(x_evaluation)

        print(predictions)

        # calcolo del reconstrution error
        RE = np.mean(np.power(x_evaluation - predictions, 2), axis=1)

        print(RE)

        # confronto  anomaly score / threshold
        outcome = RE <= self.threshold
        return outcome

    # -----------------------------------------------#
    #           instructor-provided code             #
    # -----------------------------------------------#
    def computeThreshold(self, x_thSet):

        x_thSetPredictions = self.autoencoder.predict(x_thSet)
        mse = np.mean(np.power(x_thSet - x_thSetPredictions, 2), axis=1)
        threshold = np.percentile(mse, 95)

        return threshold

    # -----------------------------------------------#
    #           instructor-provided code             #
    # -----------------------------------------------#
    def plot_reconstruction_error(self, x_evaluation, evaluationLabels):

        predictions = self.autoencoder.predict(x_evaluation)
        mse = np.mean(np.power(x_evaluation - predictions, 2), axis=1)

        trueClass = evaluationLabels != 'BENIGN'

        errors = pd.DataFrame({'reconstruction_error': mse, 'true_class': trueClass})

        groups = errors.groupby('true_class')
        fig, ax = plt.subplots(figsize=(8, 5))
        right = 0
        for name, group in groups:
            if max(group.index) > right:
                right = max(group.index)

            ax.plot(group.index, group.reconstruction_error, marker='o', ms=5, linestyle='', markeredgecolor='black',  # alpha = 0.5,
                    label='Normal' if int(name) == 0 else 'Attack', color='green' if int(name) == 0 else 'red')

        ax.hlines(self.threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors='red',
                  zorder=100, label='Threshold', linewidth=4, linestyles='dashed')
        ax.semilogy()
        ax.legend()
        plt.xlim(left=0, right=right)
        plt.title('Reconstruction error for different classes')
        plt.grid(True)
        plt.ylabel('Reconstruction error')
        plt.xlabel('Data point index')
        plt.show()
