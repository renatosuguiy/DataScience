#!/usr/bin/env python
# coding: utf-8

# # Diagnóstico de Covid usando Machine Learning 
# 
# ## Trabalho feito para a matéria de Aplicações de Multimídia da Pós-Graduação em Ciência de Dados ofertado pela PUCPR - Curutiba.
# ## Ministrado pelo professor Carlos N. Silla Jr.
# ## Elaborado por Cinthya Oestreich e Renato Suguiy
# 

# ### Os dados são extraidos da base RYDLS-20 disponibilizada pelo Prof. Silla. Ela é composta por imagens de raio-x do peito de pacientes 
# ### O scrip usa o extrator de caracteriscticas LBP e compara diferentes métodos de Machine Learning

# In[8]:


# %load LBP_feature_extractor
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd
import imghdr
import matplotlib.pyplot as plt


NRI_UNIFORM_FEATURE_NUMBER = 59

# Setting up the train and test directories
train_directory = './Raw Dataset (RYDLS-20)' #Diretorio que voces decompacataram a RYDLES.
lbp_extractor = 'nri_uniform'

# Setting up the resulting matrices directories
feature_matrix_train_path = 'Feature Matrix Train'

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    # LBP Feature Extractor from Skimage
    def describe_lbp_method_ag(self, image):
        lbpU = local_binary_pattern(image, self.numPoints, self.radius, method='nri_uniform')
        hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), normed=True)

        #Exibe no console o vetor de características:
        print(hist0)

        return hist0

# Function to load an image from a path
def open_img(filename):
    img = Image.open(filename)
    return img

# Verify if a given image is using a valid format
def verify_valid_img(path):
    possible_formats = ['png','jpg','jpeg','tiff','bmp','gif']
    if imghdr.what(path) in possible_formats:
        return True
    else:
        return False

# Feature extraction call
def feature_extraction(image, lbp_extractor):
    lbp = LocalBinaryPatterns(8, 2) #Paramametros do LBP.
    image_matrix = np.array(image.convert('L'))
    img_features = lbp.describe_lbp_method_ag(image_matrix)

    return img_features.tolist()

def create_columns(column_number, property):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append(property)
    return columns

# Function to create the training feature matrix, it has the expected class for each sample
def create_feature_matrix_train(train_directory, lbp_extractor):
    # Variable to store the data_rows
    rows_list = []

    print("Started feature extraction for the training dataset")

    # Iterate over subdirectories in training folder (1 folder for each class)
    for dir in os.listdir(train_directory):

        print("Estou em", dir);

        # This is the path to each subdirectory
        sub_directory = train_directory + '/' + dir

        # Retrieve the files for the given subdirectory
        training_filelist = os.listdir(sub_directory)

        # Iterate over all the files in the class folder
        for file in training_filelist:
            file_path = sub_directory + '/' + file

            if verify_valid_img(file_path):
                print("Processing: "+file_path)

                image = open_img(file_path)
                img_features = feature_extraction(image, lbp_extractor)

                # The name of the directory is the class
                img_features.append(dir)

                rows_list.append(img_features)
            else:
                print("The following file is not a valid image: "+file_path)

    # Creating a dataframe to store all the features
    columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER, 'class')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    print("Finished creating Training Feature Matrix")

    return feature_matrix

if not os.path.isdir(feature_matrix_train_path):
    print('Creating Directory: '+feature_matrix_train_path)
    os.mkdir(feature_matrix_train_path)

feature_matrix_train = create_feature_matrix_train(train_directory, lbp_extractor)
print("Saving Training Feature Matrix to CSV")
feature_matrix_train.to_csv(feature_matrix_train_path + '/feature_matrix_train.csv', index=False)

print('FIM')


# In[3]:


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[4]:


# Separa em dois arquivos, um com as caracteristicas das imagens e outro com a classificação de cada foto

import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd
import imghdr
import matplotlib.pyplot as plt
features_inicial = pd.read_csv("./Feature Matrix Train/feature_matrix_train.csv")
#features_inicial.head(5)
features_inicial = features_inicial
X=features_inicial.drop(["class"], axis =1)
#X.head(5)
X=X.to_numpy()
print(X.shape)

y=features_inicial["class"]
y.head(5)
y=y.to_numpy()
print(y.shape)


# In[5]:


#Reclassifica a base em 3 classes: COVID-19, Not Covid e Normal
from collections import Counter

print(Counter(y))
print(len(y))
for a in range(len(y)):
    if y[a] =="Streptococcus":
        y[a]="Not Covid"
    elif y[a] =="Pneumocystis":
        y[a]="Not Covid"
    elif y[a] =="SARS":
        y[a]="Not Covid"
    elif y[a] =="MERS":
        y[a]="Not Covid"
    elif y[a] =="Varicella":
        y[a]="Not Covid"
        
print(Counter(y))


# In[10]:


# Métodos que serão testados 

get_ipython().system('pip install deslib ')
get_ipython().system('pip install xgboost')

import numpy as np
import urllib
from sklearn.naive_bayes import GaussianNB
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as pl
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import get_cmap
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from deslib.util.datasets import make_P2
from sklearn.model_selection import GridSearchCV
from sklearn import manifold, datasets
from sklearn.manifold.t_sne import TSNE
from sklearn.datasets.base import load_digits
import itertools
import pandas as pd
import seaborn as sns
import urllib
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada

rng = np.random.RandomState(456)

# Define single classifiers

lr = LogisticRegression(C=1e5,solver='lbfgs',max_iter = 500)
dt =  DecisionTreeClassifier(criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=3)
mlp = MLPClassifier(solver='sgd', early_stopping=False, hidden_layer_sizes=(200), activation='logistic', batch_size=100, max_iter=10000, learning_rate_init=0.1, momentum=0.2, tol=1e-10, random_state=rng)
nb = GaussianNB(var_smoothing=1e-09)

# Define ensembles
rf = RandomForestClassifier(n_estimators=100, random_state=0)
#rf = RandomForestClassifier(max_features = 'auto',max_depth= 8 ,criterion ='entropy', n_estimators=200, random_state=50, oob_score = True)
xgb = XGBClassifier(  learning_rate=0.1,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=6, 
                      gamma=10,
                      min_child_weight=1)

cb = VotingClassifier(estimators=[('RandomFlorest', rf), ('XGB', xgb)], voting='soft')

# parameters for SVM
parameters = [
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['poly']},
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
]
svm = SVC(gamma='scale')
svm = GridSearchCV(svm, parameters, scoring = 'accuracy', cv=8)

titles = ['LogisticRegresson', 'DecisionTree', 'KNN', 'NaiveBayes', 'MLP', 'RF', 'RF+XGB','XGB', 'SVM']
methods = [lr, dt, knn, nb, mlp, rf, cb, xgb, svm]


# ## Treino sem Oversampling e Undersampling usando validação cruzada

# In[11]:


#Acurácia dos métodos
scores = []
for method, name in zip(methods, titles):
    folds=10
    result = model_selection.cross_val_score(method, X, y.ravel(), cv=folds,n_jobs=-1)
    scores.append(result.mean())
    print("Classification accuracy {} = {}"
          .format(name, result.mean(), result.std()))


# In[12]:


# Plotting the Confusion Matrix

import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

fig3, sub1 = plt.subplots(5, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
class_names=['COVID-19','Normal', 'Not Covid']
for clf, ax, title in zip(methods, sub1.flatten(), titles):
    y_predicted = cross_val_predict(clf, X, y, cv=10,n_jobs=-1)
    cm = confusion_matrix(y, y_predicted)
    #plot_confusion_matrix(ax, cm, title)
    df_cm = pd.DataFrame(cm, index = [i for i in "012"],
                  columns = [i for i in "012"])
    sns.heatmap(df_cm, annot=True, ax=ax, fmt = 'd')
    ax.set_title('Confusion Matrix --> ' + title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fScore = f1_score(y, y_predicted ,average=None)
    precision = precision_score(y, y_predicted ,average=None)
    recall = recall_score(y, y_predicted ,average=None)
    
    numpy_data = np.array([fScore,precision,recall]) 
    df = pd.DataFrame(data=numpy_data, index=["F1-Score", "Precisão","Recall"], columns = class_names)
    print(title )
    print(df)
    print('\n')

plt.show()
plt.tight_layout()


# ## Treinamento com Oversampling(Smote)

# In[13]:


#Oversampling usando SMOTE 
get_ipython().system('pip install imblearn')

from imblearn.over_sampling import SMOTE

strategy = {'COVID-19': 350, 'Not Covid': 350} 
oversample = SMOTE(sampling_strategy=strategy, k_neighbors=5)


# In[14]:


#Acurácia dos métodos
from imblearn.pipeline import Pipeline

scores = []
std = []
for method, name in zip(methods, titles):
    # transform the dataset
    steps = [('over', oversample),('method' ,method)]
    pipeline = Pipeline(steps=steps)
    folds=10
    result = model_selection.cross_val_score(pipeline, X, y.ravel(), cv=folds,n_jobs=-1)
    scores.append(result.mean())
    print("Classification accuracy {} = {}"
          .format(name, result.mean(), result.std()))


# In[25]:


fig3, sub1 = plt.subplots(5, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
class_names=['COVID-19','Normal', 'Not Covid']

for clf, ax, title in zip(methods, sub1.flatten(), titles):
    steps = [('over', oversample),('method' ,clf)]
    pipeline = Pipeline(steps=steps)   
    y_predicted = cross_val_predict(pipeline, X, y, cv=10,n_jobs=-1)
    cm = confusion_matrix(y, y_predicted)
    #plot_confusion_matrix(ax, cm, title)
    df_cm = pd.DataFrame(cm, index = [i for i in "012"],
                  columns = [i for i in "012"])
    sns.heatmap(df_cm, annot=True, ax=ax, fmt = 'd')
    ax.set_title('Confusion Matrix --> ' + title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fScore = f1_score(y, y_predicted ,average=None)
    precision = precision_score(y, y_predicted ,average=None)
    recall = recall_score(y, y_predicted ,average=None)
    
    numpy_data = np.array([fScore,precision,recall]) 
    df = pd.DataFrame(data=numpy_data, index=["F1-Score", "Precisão","Recall"], columns = class_names)
    print(title )
    print(df)
    print('\n')

plt.show()
plt.tight_layout()


# ## Treinamento com Oversampling (SMOTE) e Undersampling (TomekLinks)

# In[27]:


#Oversampling (SMOTE) e Undersampling (TomekLinks)

from imblearn.combine import SMOTETomek

strategy = {'COVID-19': 350, 'Not Covid': 350} 
overUnder = SMOTETomek(smote=SMOTE(sampling_strategy=strategy, k_neighbors=5))


# In[28]:


#Acurácia dos métodos
scores = []
std = []
for method, name in zip(methods, titles):
    # transform the dataset
    steps = [('overunder', overUnder),('method' ,method)]
    pipeline = Pipeline(steps=steps)
    folds=10
    result = model_selection.cross_val_score(pipeline, X, y.ravel(), cv=folds,n_jobs=-1)
    scores.append(result.mean())
    print("Classification accuracy {} = {}"
          .format(name, result.mean(), result.std()))


# In[26]:


fig3, sub1 = plt.subplots(5, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
class_names=['COVID-19','Normal', 'Not Covid']

for clf, ax, title in zip(methods, sub1.flatten(), titles):
    steps = [('overunder', overUnder),('method' ,clf)]
    pipeline = Pipeline(steps=steps)   
    y_predicted = cross_val_predict(pipeline, X, y, cv=10,n_jobs=-1)
    cm = confusion_matrix(y, y_predicted)
    #plot_confusion_matrix(ax, cm, title)
    df_cm = pd.DataFrame(cm, index = [i for i in "012"],
                  columns = [i for i in "012"])
    sns.heatmap(df_cm, annot=True, ax=ax, fmt = 'd')
    ax.set_title('Confusion Matrix --> ' + title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fScore = f1_score(y, y_predicted ,average=None)
    precision = precision_score(y, y_predicted ,average=None)
    recall = recall_score(y, y_predicted ,average=None)
    
    numpy_data = np.array([fScore,precision,recall]) 
    df = pd.DataFrame(data=numpy_data, index=["F1-Score", "Precisão","Recall"], columns = class_names)
    print(title )
    print(df)
    print('\n')

plt.show()
plt.tight_layout() 


# In[ ]:


#Previsão para imagens novas usando RF

Z = pd.read_csv('/content/drive/My Drive/Pós Ciencia de Dados/Aplicações em Multimídia/Aula 3/feature_matrix_desafio-covid-multimedia-v2.csv', header=0)
Z=Z.to_numpy()
print(Z.shape)
rf.fit(X, y) #treinando usando a base inteira 

classPredicted = rf.predict(Z)
print(classPredicted)

