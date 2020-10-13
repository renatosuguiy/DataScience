#!/usr/bin/env python
# coding: utf-8

# # **Processamento de Linguagem Natural**
# 
# ## Trabalho feito para a matéria de Processamento de Linguagem Natural da Pós-Graduação em Ciência de Dados ofertado pela PUCPR - Curitiba.
# ## Ministrado pelo professor Emerson Cabrera Paraiso.
# ## Elaborado por Renato Suguiy

# In[1]:


import tensorflow as tf


# In[1]:


#importando biblioteca de stopwords 
import nltk
nltk.download('stopwords')


# In[3]:


import csv
import io
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from string import punctuation
from sklearn.model_selection import train_test_split
#STOPWORDS = set(stopwords.words('portuguese'))

STOPWORDS = set(stopwords.words('portuguese') + list(punctuation))


# In[4]:


# Parâmetros Principais

# Tamanho do vocabulário a ser criado com o tokenizer. Este considera as N palavras mais comuns (ou frequentes nos textos)
vocab_size = 50000

# Tamanho do vetor de características que representa cada palavra. Cada palavra é transformada para um vetor com 64 entradas.
embedding_dim = 64

# Tamanho máximo da sequência de códigos que representa um texto (tamanho do feature vector)
max_length = 35

# Tipo de truncagem a ser utilizado, 'post' significa remover valores do final da sequência.
trunc_type = 'post'

# Tipo de padding a ser utilizado, 'post' significa completar no final da sequência.
padding_type = 'post'

# Token a ser utilizado quando uma palavra não presente no vocabulário é encontrada no texto
Not_known = '<NKN>'

# Porcentagem de instâncias a ser utilizada no treinamento
training_portion = .7


# In[5]:


# Carrega os textos. Em X o texto de entrada e em y os rótulos.
# Cada linha do texto carrega é filtrada para retirada de stopwords.

def ler(file):
    with open(file, 'rt', encoding='utf8') as txtfile:
        classe = ''
        positivo, negativo, neutro, surpresa = 0,0,0,0
        texto = csv.reader(txtfile, delimiter=';')      # lê um texto do arquivo
        print(texto)
        next(texto)
        for linha in texto:       # processa cada linha do texto
            #texto = linha.split(';')
            
            if linha[0] == 'surpresa':
                surpresa = surpresa + 1
                continue
            elif linha[0] == 'neutro':
                classe = 'neutro'
                neutro = neutro + 1
            elif linha[0] == 'alegria':
                positivo = positivo + 1
                classe = 'positivo'
            elif linha[0] == 'raiva' or 'medo' or 'desgosto' or 'tristeza':
                negativo = negativo + 1
                classe = 'negativo'
           
            y.append(classe)    # rotulo da linha
            aux = linha[2]
            for word in STOPWORDS:        # retirada de stopwords
                token = ' ' + word + ' '
                aux = aux.replace(token, ' ')
                aux = aux.replace(' ', ' ')
            X.append(aux)
            
        print(surpresa,positivo, negativo, neutro)
X = []
y = []
ler('2000_textos.txt')

print(len(y))       # quantidade de rótulos
print(len(X))       # quantidade de textos

train_X, validation_X, train_y, validation_y = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)


# In[6]:


# Cria o vocabulário a partir da base de treinamento considerando o tamanho definido em vocab_size.
# Utiliza como coringa o símbolo Not_known
tokenizer = Tokenizer(num_words = vocab_size, oov_token=Not_known)
tokenizer.fit_on_texts(train_X)
word_index = tokenizer.word_index


# In[7]:


# Lista os N primeiros vocábulos do dicionários (os N mais frequentes)
N=10
dict(list(word_index.items())[0:N])


# In[8]:


# Converte uma linha de texto em uma sequência de valores
train_sequences = tokenizer.texts_to_sequences(train_X)


# In[9]:


# Mostra uma linha de texto convertida para sequência de valores
# Cada valor representa uma palavra do vocabulário
print(train_sequences[5])


# In[10]:


# Transforma todas as sequências para um tamanho fixo. Sequências pequenas são completadas e sequências maiores que o limite são truncadas
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# In[11]:


print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))


# In[12]:


# Imprime uma sequência
print(train_padded[6])


# In[13]:


# Tokeniza a base de validação.

validation_sequences = tokenizer.texts_to_sequences(validation_X)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)


# In[14]:


# Mostra o conjunto de rótulos
print(set(y))


# In[15]:


# Tokeniza os rótulos
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y)

# Observe que subtraímos 1 dos códigos para estes comecem em 0
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_y))-1
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_y))-1


# In[16]:


print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[400])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[50])
print(validation_label_seq.shape)


# In[17]:


# Confere como ficaram as nossas frases depois de transformdas
# Apenas para conferência.

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_article(train_padded[10]))
print('---')
print(train_X[10])


# In[18]:


# Criando a rede LSTM (Long Short Term Memory)

hidden_size=64
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
#model.add(tf.keras.layers.LSTM(embedding_dim, dropout = 0.25 , recurrent_dropout=0.25))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.summary()
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# Compilando a LSTM
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinamento da LSTM
num_epochs = 100
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)


# In[19]:


# Criando uma RNN (Recurrent Neural Network)

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model2.add(tf.keras.layers.SimpleRNN(units=64))
model2.add(tf.keras.layers.Dense(64, activation='relu'))
model2.add(tf.keras.layers.Dense(3, activation='softmax'))
model2.summary() 

# Compilando a RNN
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinamento da RNN
history2 = model2.fit(train_padded, training_label_seq, batch_size=32, epochs=100, validation_data=(validation_padded, validation_label_seq), verbose=2)


# In[25]:


# Imprime gráfico histórico do treinamento
import matplotlib.pyplot as plt
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

plot_graphs(history2, "accuracy")
plot_graphs(history2, "loss")  


# In[21]:


# Testando uma frase

txt = ['correção: cremer pode deixar o novo mercado. na nota enviada anteriormente, em vez do título controladores da cremer devem deixar o novo mercado, o correto é cremer deve deixar o novo mercado.']
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
labels = ['positivo', 'negativo', 'neutro']
print(pred, labels[np.argmax(pred)])

print("Resultado na validacao:")
score=model.evaluate(validation_padded, validation_label_seq, verbose=0)
score2=model2.evaluate(validation_padded, validation_label_seq, verbose=0)

print('LSTM')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('RNN')
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])


# In[22]:


# Plotar a matrix de confusão 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation = 'nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)
  
  if normalize:
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    print("Normalized confusion matrix")
  else:
    print("Confusion matrix, without normalization")
  
  thresh = cm.max()*5
  for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black")
   
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


# In[23]:


# Preparar Matriz de Confusão LSTM
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


y_predict=np.argmax(model.predict(validation_padded), axis=-1)
val=validation_label_seq.reshape(len(validation_label_seq))
cm=confusion_matrix(val, y_predict)
cm_plot_labels=['negativo','neutro', 'positivo']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion matrix')
fScore = f1_score (val, y_predict,average=None)
precision = precision_score(val, y_predict,average=None)
recall = recall_score(val, y_predict,average=None)
print('\t  Negativo', ' Neutro   ', ' positivo ')
print('F-Score\t',fScore)
print('Precisão',precision)
print('Recall\t',recall)


# In[24]:


# Preparar Matriz de Confusão RNN
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


y_predict=np.argmax(model2.predict(validation_padded), axis=-1)
val=validation_label_seq.reshape(len(validation_label_seq))
cm=confusion_matrix(val, y_predict)
cm_plot_labels=['negativo','neutro', 'positivo']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion matrix')
fScore = f1_score (val, y_predict,average=None)
precision = precision_score(val, y_predict,average=None)
recall = recall_score(val, y_predict,average=None)
print('\t  Negativo', ' Neutro   ', ' positivo ')
print('F-Score\t',fScore)
print('Precisão',precision)
print('Recall\t',recall)

