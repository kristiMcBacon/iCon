"""
@author: Gruppo DASHAJ-MARINELLI

Questo script analizza un dataset sull'autismo utilizzando il classificatore SVM.
Viene effettuato il caricamento dei dati, la loro elaborazione, l'addestramento del modello,
e l'analisi delle sue prestazioni attraverso metriche e grafici.
"""
import numpy as np, pandas as pd, seaborn as sn
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size = 0.7, random_state = 13)

#-------------------scelta valore gamma-----------------------------------
# Definisco i valori di gamma da testare
param_grid = {'gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]}

# Creazione del modello SVC
svm_model = svm.SVC()

# Ricerca a griglia con cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X1_train, y1_train)

# Accesso ai risultati della ricerca a griglia
results = grid_search.cv_results_

# Stampa dei valori di gamma testati e delle prestazioni corrispondenti
for gamma, mean_score in zip(results['param_gamma'], results['mean_test_score']):
    print(f"Gamma: {gamma:<10}Mean Score: {mean_score}")

    
# Stampa il valore di gamma migliore
print("\nMiglior valore di gamma:", grid_search.best_params_['gamma'])
#-------------------------------------------------------------------------

# Addestramento del classificatore SVM
clf = svm.SVC(kernel='rbf', gamma=0.01)
clf.fit(X1_train, y1_train) 

# Effettua previsioni sul test set
prediction = clf.predict(X1_test)
accuracy = accuracy_score(prediction, y1_test)
print ('\naccuracy_score:',accuracy)

# Stampa del rapporto di classificazione e della matrice di confusione
print ('\nClasification report:\n',classification_report(y1_test, prediction))
print ('\nConfussion matrix:\n',confusion_matrix(y1_test, prediction))

# Creazione e visualizzazione della matrice di confusione come heatmap
confusion_matrix = confusion_matrix(y1_test, prediction)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

# Valutazione del modello attraverso cross-validation (di 5)
cv_scores = cross_val_score(clf, X1, y1, cv=5)

# Stampa delle statistiche ottenute dalla cross-validation
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))
print('\n')

# Calcolo dell'average precision e visualizzazione della curva precision-recall
average_precision = average_precision_score(y1_test, prediction)
precision, recall, _ = precision_recall_curve(y1_test, prediction)

# In Matplotlib < 1.5, plt.fill_between non ha l'argomento 'step'
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

# Calcolo del punteggio F1
f1 = f1_score(y1_test, prediction)

# Creazione di un grafico a barre per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()