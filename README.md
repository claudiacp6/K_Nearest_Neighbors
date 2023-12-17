#  K-Nearest Neighbors (kNN)

O KNN, ou "k-Nearest Neighbors", é um algoritmo utilizado tanto para classificação como para regressão. Ele é conhecido como um método de aprendizado baseado em instância ou método preguiçoso (lazy) porque não tenta construir um modelo explícito durante a fase de treinamento. Em vez disso, ele memoriza os exemplos de treinamento e faz previsões com base na proximidade (similaridade) dos novos exemplos aos exemplos existentes.

A ideia básica do k-NN é que os exemplos em um espaço de características semelhantes tendem a ter rótulos semelhantes. Ou seja, ao prever o rótulo de um novo exemplo, o algoritmo olha para os k exemplos de treinamento mais próximos a ele no espaço de características e atribui ao novo exemplo a classe mais comum entre esses k vizinhos.  

Este trabalho desenvolve um código de Python "from scratch" e está dividido em 4 partes.

## *1. Código de inicialização na classe KNN*

       def __init__(self, k=3, normalize=False):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
   
Com a função __init__ são definidos 3 parâmetros: 
  - **k**: Número de vizinhos mais próximos a serem considerados pelo algoritmo k-NN (o padrão é 3).
  - **Normalize**: Um bool que indica se os dados devem ser normalizados ou não (o padrão é False).
  - **Atributos**: O método __init__ inicializa alguns atributos da instância da classe com valores None.

## *2. Função fit*
  É um método que é comumente usado em algoritmos de aprendizado de máquina para treinar o modelo com base nos dados de treinamento fornecidos. No contexto do k-NN (k-Nearest Neighbors), o método fit é responsável por armazenar os dados de treinamento na instância do objeto para que possam ser usados posteriormente para fazer previsões.
   
       def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
   
  Onde:
   - **x_train**: É o conjunto de dados de treinamento, contendo as características (features) dos exemplos de treinamento. Cada linha representa um exemplo, e cada coluna representa uma característica.
   - **y_train**: São os rótulos correspondentes aos exemplos de treinamento em X_train. Cada rótulo é associado a um exemplo específico e indica a classe ou o valor que o modelo deve aprender.

Dentro do método fit, os dados de treinamento (X_train e y_train) são simplesmente armazenados nos atributos da instância self.X_train e self.y_train. Isso significa que, após chamar o método fit, a instância da classe terá acesso aos dados de treinamento, e o modelo poderá ser treinado com base nesses dados.

## *3. Método Predict*
  É responsável por fazer previsões com base nos dados de teste fornecidos. 
    - X_test: São os dados de teste nos quais serão feitas as previsões.

          def predict(self, X_test):
            if self.normalize:
             X_test = self.normalize_data(X_test)
          predictions = []
          for x in X_test:
             distances = np.linalg.norm(self.X_train - x, axis=1)
             indices = np.argsort(distances)[:self.k]
             k_nearest_labels = self.y_train[indices]
             unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
             predicted_label = unique_labels[np.argmax(counts)]
             predictions.append(predicted_label)
          return np.array(predictions)
        
Código utilizado contém:
   - **Normalização** (se necessário): Se o atributo normalize for True, os dados de teste (X_test) são normalizados usando a função normalize_data antes de fazer previsões. Isso garante que os dados de teste sejam processados da mesma maneira que os dados de treinamento.

   - **Cálculo das Distâncias**: Para cada exemplo em X_test, a distância é calculada em relação a todos os exemplos de treinamento em X_train. A função np.linalg.norm é usada para calcular a distância euclidiana.

   - **Identificação dos Vizinhos Mais Próximos**: Os índices dos k vizinhos mais próximos são obtidos usando argsort para ordenar as distâncias e selecionar os k primeiros índices.

   - **Contagem dos Rótulos**: Conta-se a ocorrência de cada rótulo nos k vizinhos mais próximos.

   - **Atribuição do Rótulo Predito**: O rótulo predito é determinado pelo rótulo mais comum entre os k vizinhos mais próximos.

   - **Armazenamento das Previsões**: As previsões são armazenadas na lista predictions.

   - **Retorno das Previsões**: As previsões são retornadas como um array NumPy.

## *4. Normalização dos dados*
  É utilizado para normalizar os dados, se a opção de normalização estiver ativada (self.normalize == True). Normalização é um processo comum em algoritmos de Machine Learning, onde os valores das características (features) são ajustados para garantir que estejam na mesma escala.

       def normalize_data(self, data):
         min_vals = np.min(data, axis=0)
         max_vals = np.max(data, axis=0)
         return (data - min_vals) / (max_vals - min_vals)

  Normalização dos dados incluí:
   - **Cálculo dos Mínimos e Máximos**: Para cada coluna (característica) em data, np.min e np.max são usados para calcular os valores mínimos e máximos.

   - **Normalização**: Cada valor em data é normalizado subtraindo o valor mínimo da coluna correspondente e dividindo pelo intervalo (diferença entre o máximo e o mínimo).

  - **Retorno dos Dados Normalizados**: Os dados normalizados são retornados pela função.

Esse método é utilizado no método predict antes de fazer previsões, caso a opção de normalização (self.normalize) esteja ativada.

# Exemplo de Aplicação Prática

Neste exemplo serão criados dados exemplo para um problema de classificação binária. 
São gerados números aleatórios pela função np.random-seed(42). Posteriormente é gerada uma matriz de números entre 0 e 1, com dimensões 100x2, onde cada linha representa um exemplo e cada coluna representa uma caracterísitica.




       np.random.seed(42)
       X_train = np.random.rand(100, 2)
       y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

       X_test = np.random.rand(20, 2)
       y_test = (X_test[:, 0] + X_test[:, 1] > 1).astype(int)


*y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)*: Calcula a soma dos elementos em cada linha de X_train e verifica se essa soma é maior que 1. O resultado é um array booleano, que é então convertido para inteiros (0 ou 1) usando astype(int). Isso cria os rótulos de classe binária com base na condição mencionada.

O modelo com os dados criados é treinado com a função *fit* e as previsões são geradas e armazenadas na variável predictions, com K=3.

       knn = KNN(k=3)
       knn.fit(X_train, y_train)

       predictions = knn.predict(X_test)

Para a visualização gráfica: 


       plt.figure(figsize=(10, 6))

       plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolor='k', marker='o', label='Treinamento')

       correct_predictions = predictions == y_test
       incorrect_predictions = ~correct_predictions

       plt.scatter(X_test[correct_predictions, 0], X_test[correct_predictions, 1], c='green', marker='^', s=100, label='Previsão Correta')
       plt.scatter(X_test[incorrect_predictions, 0], X_test[incorrect_predictions, 1], c='red', marker='v', s=100, label='Previsão Incorreta')


Em resumo, este código cria uma visualização que mostra os pontos de treino coloridos de acordo com as classes, e os pontos de teste são marcados de forma diferente dependendo se as previsões do modelo foram corretas ou incorretas. A legenda ajuda a distinguir entre os pontos de previsões corretas e incorretas. Isso é útil para avaliar visualmente o desempenho do modelo k-NN nos dados de teste.

![image](https://github.com/claudiacp6/K_Nearest_Neighbors/assets/147619731/4a7090dd-e8f7-4fad-8309-29c3acf27415)




