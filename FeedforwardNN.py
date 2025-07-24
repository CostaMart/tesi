#
#       Feedforward neural network for binary classification -
#
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import keras.utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import initializers, regularizers
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model

mpl.use('TkAgg')
tf.config.experimental.enable_op_determinism()
keras.utils.set_random_seed(1)


class FeedforwardNN():

    def __init__(self, input_dim, l1size, l2size, activation, dropout_rate):
        input_layer = Input(shape=(input_dim, ))

        layer = Dense(l1size, activation=activation,
                      kernel_initializer=initializers.RandomNormal(stddev=0.01))(input_layer)
        dropout = Dropout(dropout_rate)(layer)
        layer = Dense(l2size, activation=activation,
                      kernel_initializer=initializers.RandomNormal(stddev=0.01))(dropout)
        dropout = Dropout(dropout_rate)(layer)
        layer = Dense(2, activation=activation,
                      kernel_initializer=initializers.RandomNormal(stddev=0.01))(dropout)
        # 3. Output layer:
        output_layer = Activation(activation='softmax')(layer)

        # Classe Model serve per definire il modello dopo aver definito i layer
        self.classifier = Model(input_layer, output_layer)

    def summary(self, ):
        self.classifier.summary()

    def train(self, x, y, epochs, batch_size):

        validation_split = 0.1  # usiamo il 10% del training set per la validazione

        # Settiamo ottimizzatore e loss function
        self.classifier.compile(optimizer='Adam', loss='categorical_crossentropy')
        history = self.classifier.fit(x,
                                      y,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      shuffle=True,  # lo shuffle serve per mescolare i dati
                                      verbose=2
                                      )
        # history ci da la storia dell'addestramento per vedere come sta andando

        #

        # -----------------------------------------------#
        #           instructor-provided code            #
        # -----------------------------------------------#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

        df_history = pd.DataFrame(history.history)
        return history, df_history

    # Operaimo sulla prediction per fare la validazione della rete neurale.
    # Qui passiamo solo le x, non le label di valutazione
    def predict(self, x_evaluation):
        # Ottengo le predizioni con sel.classifier.predict
        # prende una matrice di ingresso e la interpreta riga per riga
        prediction = self.classifier.predict(x_evaluation)
        outcome = prediction[:, 0] > prediction[:, 1]
        return outcome
       # print(prediction)
