"""
@author: Gruppo DASHAJ-MARINELLI

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score

dataset0 = pd.read_csv('Ontologia/Autism-Dataset.csv')
dataset = dataset0
 # applicazione del one-hot encoding sulle feature di tipo stringa
categorical_features = ["age","ethnicity","screening_score","contry_of_res","test_compiler"]

onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_dataset = onehot_encoder.fit_transform(dataset[categorical_features])

# combinazione delle feature encodate con il resto del dataset
encoded_dataset = pd.concat([dataset.drop(columns=categorical_features), pd.DataFrame(encoded_dataset)], axis=1)

# esecuzione K-means per un range di K cluster
dataset = np.array(encoded_dataset)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=40)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)  

# tracciamento del grafico
plt.plot(range(1, 11), wcss, 'bx-')
plt.title('elbow method')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('WCSS')
plt.show() 

kmeans = KMeans(n_clusters=2, n_init=10, random_state=40)
kmeans.fit(dataset)

# Calcolo della somma dei quadrati intra-cluster (WCSS)
wcss = kmeans.inertia_
print("WCSS:", wcss)

# Calcolo dello Silhouette Score
silhouette_avg = silhouette_score(dataset, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)
cluster_result = kmeans.labels_

dataset0['cluster'] = cluster_result

# Riordina le colonne del DataFrame
columns_order = list(dataset0.columns)
columns_order.remove('Class/ASD')
columns_order.append('Class/ASD')
columns_order.remove('cluster')
columns_order.insert(-1, 'cluster')
dataset_reordered = dataset0[columns_order]

dataset_reordered.to_csv('Clustering/Autism-Dataset+clusters.csv', index=False)