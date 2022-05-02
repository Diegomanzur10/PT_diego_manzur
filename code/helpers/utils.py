import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

import pickle
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, save_model
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
import tensorflow_addons as tfa
import tensorflow

# A continuación se definen las métricas de ajuste
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def load_data_lstm(X, seq_len, train_size=(109/115)):
    # Esta función nos permite separar y organizar los datos en train dataset y test dataset.
    # El valor train_size=109/105 equivale a que la base de datos de test tenga 60 datos, que es el benchmark del problema.
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    data = []
    
    for index in range(len(X_mat) - sequence_length):
        data.append(X_mat[index: index + sequence_length])
    
    data = np.array(data)
    train_split = int(round(train_size * data.shape[0]))
    train_data = data[:train_split, :]
    
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1][:,-1]
    
    x_test = data[train_split:, :-1] 
    y_test = data[train_split:, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return x_train, y_train, x_test, y_test



# A continuación definimos 3 tipos de modelos. Dado que tenemos que definir para la red neuronal varios parámetros, entre ellos
# la función de activación, la tasa de aprendizaje, el número de capaz y el número de neuronas por capa, 
# así como la función de costo a minimizar, multiples opciones podemos crear. Para este caso, además nos apoyamos con
# dos algorítmos de optimizacion (Adam y Dropout) para obtener resultados numéricos mas estables.

def build_model1(input_shape):
    d = 0.2 #Parámetro de Dropout, la literatura recomienda un valor cercano a 0.2
    model = Sequential()
    
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True)) # El número de neuronas (128) asegura encontrar
    model.add(Dropout(d))                                                # patrones en la serie de tiempo 
        
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))  # Función de activación ideal para capas ocultas      
    model.add(Dense(1,kernel_initializer="uniform",activation='linear')) # La literatura recomienda esta función para la capa de
                                                                         # salida
    
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'], )
    return model
def build_model2(input_shape):
    d = 0.2
    model = Sequential()
    
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='sigmoid'))
    
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model

def build_model3(input_shape):
    d = 0.2
    model = Sequential()
    
    model.add(LSTM(1024, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    model.compile(optimizer = tfa.optimizers.Lookahead(tensorflow.optimizers.Adam(0.001), sync_period = 5), 
                  loss = 'binary_crossentropy') # Tasa de aprendizaje y número de periodos de actualización de pesos "lentos",
                                                # propuesto por Michael R. Zhang et.al  Lookahead Optimizer: k steps forward, 1 step back
    return model


def media_e2(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    corr, _ = pearsonr(actual, pred)
    return (mean_actual - mean_pred)**2 + (sd_actual - sd_pred)**2 + 2*(1-corr)*sd_actual*sd_pred

def PSM(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(pred)
    return ((mean_actual - mean_pred)**2) / media_e2(actual, pred)

def PSV(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    return ((sd_actual - sd_pred)**2) / media_e2(actual, pred)

def PC_(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    corr, _ = pearsonr(actual, pred)
    return (2*(1-corr)*sd_actual*sd_pred) / media_e2(actual, pred)

