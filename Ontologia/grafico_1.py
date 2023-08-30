import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dei dati
data = pd.read_csv("Autism-Dataset.csv")

# Seleziona le colonne desiderate per le domande e is_autistic
selected_columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "is_autistic"]
# Filtra il dataset con le colonne selezionate
filtered_data = data[selected_columns]

# Calcola le medie delle risposte per i gruppi "is_autistic" 0 e 1
mean_responses = filtered_data.groupby("is_autistic").mean()

# Crea un grafico a barre per le differenze nelle medie delle risposte
plt.figure(figsize=(10, 6))
ax = mean_responses.T.plot(kind="bar", color=["skyblue", "orange"])
plt.title("Differenze nelle risposte tra gruppo is_autistic 0 e 1")
plt.xlabel("Domande")
plt.ylabel("Media delle risposte")
plt.xticks(rotation=0)

# Stampa i valori sulle colonne
for container in ax.containers:
    ax.bar_label(container, fmt='%2.2f', label_type='edge', color='black')

plt.legend(["Non Autistico (0)", "Autistico (1)"])
plt.show()