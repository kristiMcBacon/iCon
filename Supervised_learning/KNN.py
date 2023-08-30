"""
@author: Gruppo DASHAJ-MARINELLI

Questo script analizza un dataset sull'autismo utilizzando il classificatore dei k-nn.
Viene effettuato il caricamento dei dati, la loro elaborazione, l'addestramento del modello,
e l'analisi delle sue prestazioni attraverso metriche e grafici.
"""

import numpy as np, pandas as pd, seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Definizione delle feature e delle feature dummificate
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
feature_dummied = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent"]

# Caricamento del dataset da un file CSV
dataset = pd.read_csv("Ontologia/Autism-Dataset.csv", sep=",", names=feature, dtype={'A1_Score':object,'A2_Score':object,'A3_Score':object,'A4_Score':object,'A5_Score':object,'A6_Score':object,'A7_Score':object,'A8_Score':object,'A9_Score':object,'A10_Score':object,'age':object,'gender':object,'ethnicity':object,'jundice':object,'is_autistic':object,'screening_score':object,'PDD_parent':object,'Class/ASD':object})

# Dummificazione delle feature categoriche
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
data_dummies = data_dummies.drop(["Class/ASD"], axis=1)

# Preparazione dei dati di input (X) e target (y)
X = data_dummies
y = pd.get_dummies(dataset["Class/ASD"], columns=["Class/ASD"])
y = y["1"]

# Applicazione di SMOTE per bilanciare le classi
oversample = SMOTE()
X1, y1 = oversample.fit_resample (X, y)

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 13)

#------------------------------------------------------------------------------------------------------------------------------
error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#Grafico che mostra l'errore medio nelle predizioni a seguito di una variazione del valore K(numero vicini)
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize = 10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
plt.show()
#------------------------------------------------------------------------------------------------------------------------------

# Addestramento del modello con il valore ottimale di K
neigh = KNeighborsClassifier(n_neighbors = 11)
knn = neigh.fit(X_train, y_train) 

# Effettua previsioni sul test set
prediction = knn.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print(f"accuracy_score: {accuracy:.2f}")

# Stampa del rapporto di classificazione e della matrice di confusione
print ('\nClasification report:\n',classification_report(y_test, prediction))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, prediction))

# Creazione e visualizzazione della matrice di confusione come heatmap
confusion_matrix = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

# Valutazione del modello attraverso cross-validation (di 5) 
cv_scores = cross_val_score(neigh, X, y, cv=5)

# Stampa delle statistiche ottenute dalla cross-validation
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()

# Calcolo delle probabilità e dell'AUC per la curva ROC
probs = knn.predict_proba(X_test)
probs = probs[:, 1]

# Calcolo e visualizzazione della curva ROC
auc = roc_auc_score(y_test, probs)
print('\nAUC: %.3f' % auc)
# calcolo roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()

# Calcolo dell'average precision e visualizzazione della curva precision-recall
average_precision = average_precision_score(y_test, prediction)
precision, recall, _ = precision_recall_curve(y_test, prediction)

# In matplotlib < 1.5, plt.fill_betwee non dispone dell'argomento 'step'
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Calcolo dell'F1-score
f1= f1_score(y_test, prediction)

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()