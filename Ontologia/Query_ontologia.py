from owlready2 import *

print("ONTOLOGIA\n")
onto = get_ontology("ontologia.owl").load()

#stampa il contenuto principale dell'ontologia
print("-----------------------Class list in ontology:--------------------------\n")
print(list(onto.classes()), "\n")

#stampa le proprietà dell'oggetto
print("-----------------------Object property in ontology:---------------------\n")
print(list(onto.object_properties()), "\n")

#stampa le proprietà dei dati
print("-----------------------Data property in ontology:-----------------------\n")
print(list(onto.data_properties()), "\n")

#stampa gli individui della classe paziente
print("-----------------------paziente list in ontology:-----------------------\n")
paziente = onto.search(is_a = onto.paziente)
print(paziente, "\n")

#stampa gli individui della classe test
print("-----------------------test list in ontology:---------------------------\n")
test = onto.search(is_a = onto.test)
print(test, "\n")

#stampa gli individui della classe domanda
print("-----------------------domanda list in ontology:------------------------\n")
domanda = onto.search(is_a = onto.domanda)
print(domanda, "\n")

# QUERY------------------------------------------------------------------------
print("_____________________________QUERY______________________________________\n")
# Esegui la query per ottenere i pazienti che hanno effettuato il test
pazientiConTest = onto.search(is_a = onto.paziente, haTest = onto.Test)

print("- Lista dei pazienti che hanno effettuato un test:\n")
for paziente in pazientiConTest:
    print(paziente)
    
# Query per estrarre i pazienti con isAutistic impostato a True
query_result = list(onto.search(type=onto.paziente, isAutistic=True))

# Stampa i risultati
print("\n\n- Pazienti che hanno effettuato un test e sono autistici:\n")
for paziente in query_result:
    print(paziente.name)