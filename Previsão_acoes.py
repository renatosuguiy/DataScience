#!/usr/bin/env python
# coding: utf-8

# ## Copyright 2019 The TensorFlow Authors.
# @Adaptado por Renato Suguiy
# 
# 
# ---
# 
# 

# # **Previsão do Valor de Ações usando Recurrent Neural Network**
# 
# ## Trabalho feito para a matéria de Deep Learning da Pós-Graduação em Ciência de Dados ofertado pela PUCPR - Curutiba.
# ## Ministrado pelo professor Alceu de Souza Britto Jr.
# ## Elaborado por Renato Suguiy

# In[5]:


from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# In[6]:


#Carrega base da dados direto do Driver e nomeia as colunas
csv_path='/content/drive/My Drive/Pós Ciencia de Dados/Deep Learning/TRABALHO DEEP LEARNING/Trabalho2/Base de Ações Trabalho.xlsx' 
df = pd.read_excel(csv_path, header= 1, names=['dataIBOV', 'valorIBOV','nulo1','dataPetro4', 'valorPetro4','nulo2','dataVale3', 'valorVale3', 'nulo3','dataDolar', 'valorDolar'])


# Visualização dos Dados

# In[8]:


df.head(12)


# As variáveis abaixo garantem padronização e reprodutibilidade

# In[9]:


TRAIN_SPLIT = 480


# In[10]:


tf.random.set_seed(13)


# ## Part 1: Previsão univalorada
# Criando modelo utilizando apenas uma *feature* (ação). 
# 

# In[11]:


data = df['valorPetro4']

data.index = df['dataPetro4']
data = data.dropna() #Tira valores nulos

print ("Numero de Amostras: ", len(data))
print ("Vetor de valores:" , data.values)
data.head()


# Abaixo plotamos o gráfico de variação da ação até  2020

# In[12]:


data.plot(subplots=True, color='gray')


# Recuperando apenas o valores tabela e normalizando os dados
# * Importante (Apenas utilizando os dados de treinamento)
# 
# 

# In[13]:


uni_data = data.values
print ("Dados: ", uni_data)
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
print ("Média: ", uni_train_mean)
uni_train_std = uni_data[:TRAIN_SPLIT].std()
print ("Desv. Padrão: ", uni_train_std)
uni_data = (uni_data-uni_train_mean)/uni_train_std
print ("Dados Norm: ", uni_data)


# **Criação dos Datasets de Treinamento e Validação**'

# * A função abaixo retorna a fração do dataset a ser utilizada sendo:
# 
#           * history_size: janela a ser observada
#           * target_size: O exato momento a ser avaliado 
# 

# In[14]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


# In[15]:


'''Tamanho da Janela do Historico'''
univariate_past_history = 30  #n observacoes anteriores
future = univariate_future_target = 18 #n proxima observação 

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
print (x_train_uni)


# Para efeito de visualização, analisamos as janelas de observações criadas e seu respectivo preço (label)

# In[16]:


def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'gX', 'ro']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
      future = delta
    else:
      future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

sample_id = 1
print("#Amostras:", len(x_train_uni),"#Labels: ", len(y_train_uni))
print ("Amostra[0]:\n", x_train_uni[sample_id],"\nPreço: ", y_train_uni[sample_id])
show_plot([x_train_uni[sample_id], y_train_uni[sample_id]], future, 'Sample Example')


# ### Baseline
# Para efeitos comparativos, vamos criar um modelo de predição utilizando a média das últimas 20 observações. Este simples modelo nos revela como a média pode ser falha para prever séries temporais

# In[17]:


def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[sample_id], y_train_uni[sample_id], baseline(x_train_uni[sample_id])], 0,
           'Baseline Prediction Example')


# ### Recurrent neural network (SimpleRNN and LSTM)
# 
# Definindo os datasets

# In[18]:


BATCH_SIZE = 50
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# ## Criando as arquiteturas

# RNN

# In[19]:


simple_rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(8, input_shape=(x_train_uni.shape[1], 
    x_train_uni.shape[2])),    
    tf.keras.layers.Dense(1)
])


simple_rnn_model.compile(optimizer='adam', loss='mae')

simple_rnn_model.summary()


# LSTM
# 

# In[20]:


simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(x_train_uni.shape[1], 
    x_train_uni.shape[2])),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')


simple_lstm_model.summary()


# ## Treinamento
# 
# Para poupar tempo, cada epoca conterá apenas 500 amostras ao invés da base toda. Podemos alterar este número depois para ver o impacto.

# In[21]:


EVALUATION_INTERVAL = 500
EPOCHS = 25
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# RNN

# In[22]:


rnn_log = simple_rnn_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50, callbacks=[es])


# LSTM

# In[23]:


lstm_log = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50,callbacks=[es])


# ## Visualização do Treinamento

# In[24]:


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


# In[25]:


plot_train_history(rnn_log,
                  'RNN Training and validation loss')
plot_train_history(lstm_log,
                  'LSTM Training and validation loss')


# In[26]:


def plot_preds(plot_data, delta=0):
    labels = ['History', 'True Future', 'RNN Prediction','LSTM Prediction']
    marker = ['.-', 'gX', 'ro' , 'bo']
    time_steps = create_time_steps(plot_data[0].shape[0])
    

    future = delta

    plt.title('Predictions')
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

for x, y in val_univariate.take(5):
  plot = plot_preds([x[0].numpy(), y[0].numpy(),
                    simple_rnn_model.predict(x)[0], simple_lstm_model.predict(x)[0]], future)
  plot.show()


# In[27]:


#Calculs uma taxa de erro

err_rnn=0
err_lstm=0

for x, y in val_univariate.take(100):
  err_rnn += abs(y[0].numpy() - simple_rnn_model.predict(x)[0])
  err_lstm += abs(y[0].numpy() - simple_lstm_model.predict(x)[0])
  
err_rnn = err_rnn/100
err_lstm = err_lstm/100
  
print(err_rnn)
print(err_lstm)
  


# In[28]:


# Target em n dias 

ultData = np.reshape(uni_data[-1*univariate_past_history:],(1,univariate_past_history,1))

prevRnn = (simple_rnn_model.predict(ultData))* uni_train_std + uni_train_mean
prevLstm = (simple_lstm_model.predict(ultData))* uni_train_std + uni_train_mean


print('Previsão para {} dias\nRNN: {}\nLSTM: {}'.format(future,float(prevRnn[0]), float(prevLstm[0])))


# In[29]:


#Salvar o modelo 

simple_lstm_model.save('model') #pesos

jsmodel=simple_lstm_model.to_json()  #jsom
with open('model_config.json', 'w') as json_file:
    json_file.write(jsmodel)

