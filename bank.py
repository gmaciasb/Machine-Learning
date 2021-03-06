# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:54:19 2021

@author: HOMEWORK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

url = "bank.csv"
data = pd.read_csv(url)

# Tratamiento de datos

data.housing.replace(['yes', 'no'], [1, 0], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.vc.replace(['no', 'yes'], [0, 1], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data['marital'].replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data['education'].replace(['secondary', 'tertiary', 'primary', 'unknown'], [0, 1, 2, 3], inplace=True)
data['job'].replace(['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
data['month'].replace(['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'feb', 'jan', 'oct', 'sep', 'mar', 'dec'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
data['contact'].replace(['cellular', 'unknown', 'telephone'], [0, 1, 2], inplace=True)
data['poutcome'].replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)

data.drop(['campaign'], axis = 1, inplace=True)



rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)

data_train = data[:850]
data_test = data[850:]

x = np.array(data_train.drop(['vc'], 1))
y = np.array(data_train.vc)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['vc'], 1))
y_test_out = np.array(data_test.vc)

def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# Regresi??n Log??stion
print('*' * 50)
print('Regresi??n Log??stion')
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
logreg.fit(x_train, y_train)

# Accuracy de entrenamiento
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
print('*' * 50)

# Accuracy de validaci??n
y_pred = logreg.predict(x_test_out);
print(f'Accuracy de Validaci??n: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusi??n')
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)

probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)


# Regresi??n Log??stica con validaci??n cruzada
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train],y_train[train])
    scores = logreg.score(x_train[test], y_train[test])
    cvscores.append(scores)
# print(np.mean(cvscores))
# print(cvscores)

print('Regresi??n Log??stion validaci??n cruzada')
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
print(f'Accuracy de Validaci??n: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusi??n')
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)

probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)


###############################################################################
# Maquinas de Soporte Vectorial

svc = SVC(gamma = 'auto')
svc.fit(x_train, y_train)

print('Maquina de soporte Vectorial')
print(f'Accuracy de Entrenamiento: {svc.score(x_train, y_train)}')
print(f'Accuracy de Test: {svc.score(x_test, y_test)}')
print('*'* 50)

##############################################################################
# Vecinos mas cercanos clasificador

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

print('Clasificador Vecinos mas cercanos')
print(f'Accuracy de Entrenamiento: {knn.score(x_train, y_train)}')
print(f'Accuracy de Test: {knn.score(x_test, y_test)}')
print('*'* 50)

##############################################################################