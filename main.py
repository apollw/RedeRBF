import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from numpy.random import seed


class RBFNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, sigma=1.0):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.sigma = sigma

        # Inicialização dos centros das funções de base radial (RBF)
        self.centers = np.random.rand(num_hidden, num_inputs)

        # Inicialização dos pesos da camada de saída
        self.weights = np.random.rand(num_hidden, num_outputs)

    def gaussian(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def calculate_hidden_layer(self, inputs):
        # Calcula a saída da camada oculta
        hidden_layer_output = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            hidden_layer_output[i] = self.gaussian(inputs, self.centers[i], self.sigma)
        return hidden_layer_output

    def train(self, inputs, targets, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # Propagação para frente
                hidden_output = self.calculate_hidden_layer(inputs[i])
                final_output = np.dot(hidden_output, self.weights)

                # Cálculo do erro
                error = targets[i] - final_output

                # Atualização dos pesos da camada de saída
                self.weights += learning_rate * error * hidden_output.reshape(-1, 1)

    def predict(self, inputs):
        # Faz uma previsão para novos dados
        hidden_output = self.calculate_hidden_layer(inputs)
        final_output = np.dot(hidden_output, self.weights)
        return final_output


# Carregando o conjunto de dados do vinho usando o pandas
dataset_path = r"C:\Users\Leanderson\Downloads\wine.csv"
df = pd.read_csv(dataset_path)

print("CABEÇALHO DO DATASET")
df.head()

print("TIPOS DE VINHO POR CLASSES")
print(df["Wine"].unique())

print("CARACTERÍSTICAS DE DEFINEM OS TIPOS DE VINHO")
feature_names = df.columns[1:].tolist()
print(feature_names)

# Separando as características (X) e a coluna de classes (y)
X_subset = df.drop("Wine", axis=1)  # Exclui a coluna "Wine" para obter as características
y_subset = df["Wine"]  # Coluna "Wine" é o alvo

# Particionando o conjunto de dados em treinamento (70%) e teste (30%)
X_train_subset, _, y_train_subset, _ = train_test_split(X_subset, y_subset, test_size=0.3, random_state=seed())

# Normalizando os dados
X_train_normalized_subset = (X_train_subset - X_train_subset.mean()) / X_train_subset.std()

# Convertendo as classes para representação binária
y_train_binary_subset = np.where(y_train_subset == 1, 1, 0)

# Criando e treinando a rede RBF
rbf_network_wine_subset = RBFNetwork(
    num_inputs=X_train_normalized_subset.shape[1], num_hidden=4, num_outputs=1, sigma=1.0
)
rbf_network_wine_subset.train(X_train_normalized_subset.values, y_train_binary_subset, learning_rate=0.01, epochs=1000)

# Testando a rede treinada
predictions_wine_train_subset = np.array([rbf_network_wine_subset.predict(x) for x in X_train_normalized_subset.values])

# Convertendo as saídas para classes binárias
predicted_classes_wine_train_subset = np.where(predictions_wine_train_subset >= 0.5, 1, 0)

# Exibindo alguns exemplos de teste
num_examples_to_show = 5

for i in range(num_examples_to_show):
    example_idx = np.random.randint(0, len(X_train_normalized_subset))
    input_example = X_train_normalized_subset.iloc[example_idx].values
    true_label = y_train_binary_subset[example_idx]
    predicted_label = predicted_classes_wine_train_subset[example_idx]

    print(f"Exemplo {i + 1}:")
    print(f"Entrada: {input_example}")
    print(f"Rótulo Verdadeiro: {true_label}")
    print(f"Rótulo Previsto: {predicted_label}")
    print()


# Avaliando o desempenho usando métricas
accuracy_wine_train_subset = accuracy_score(y_train_binary_subset, predicted_classes_wine_train_subset)
precision_wine_train_subset = precision_score(y_train_binary_subset, predicted_classes_wine_train_subset)
recall_wine_train_subset = recall_score(y_train_binary_subset, predicted_classes_wine_train_subset)
f1_wine_train_subset = f1_score(y_train_binary_subset, predicted_classes_wine_train_subset)

# Imprimindo as métricas para o conjunto de treinamento
print("Métricas para o Conjunto de Treinamento (70%):")
print(f"Accuracy: {accuracy_wine_train_subset:.4f}")
print(f"Precision: {precision_wine_train_subset:.4f}")
print(f"Recall: {recall_wine_train_subset:.4f}")
print(f"F1 Score: {f1_wine_train_subset:.4f}")
