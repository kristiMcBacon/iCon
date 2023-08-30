import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dei dati
data = pd.read_csv("Autism-Dataset.csv")

# Seleziona le colonne desiderate per le domande e is_autistic
selected_columns = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","used_app_before","screening_score","test_compiler","PDD_parent","Class/ASD"]
# Filtra il dataset con le colonne selezionate
filtered_data = data[selected_columns]

# Creazione della matrice di correlazione
corr_matrix = filtered_data.corr()

# Creazione della heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heat Map of Dataset")
plt.show()