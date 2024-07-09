import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Cargar el dataset
wine = load_wine()
X = wine.data
y = wine.target

# Crear un DataFrame
data = pd.DataFrame(X, columns=wine.feature_names)
data['target'] = y

# Dividir el dataset en conjunto de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Guardar los conjuntos en archivos CSV
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
