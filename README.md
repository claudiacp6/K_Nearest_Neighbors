K-Nearest Neighbors (kNN)

O KNN, ou "k-Nearest Neighbors", é um algoritmo utilizado tanto para classificação como para regressão. Ele é conhecido como um método de aprendizado baseado em instância ou método preguiçoso (lazy) porque não tenta construir um modelo explícito durante a fase de treinamento. Em vez disso, ele memoriza os exemplos de treinamento e faz previsões com base na proximidade (similaridade) dos novos exemplos aos exemplos existentes.

A ideia básica do k-NN é que os exemplos em um espaço de características semelhantes tendem a ter rótulos semelhantes. Ou seja, ao prever o rótulo de um novo exemplo, o algoritmo olha para os k exemplos de treinamento mais próximos a ele no espaço de características e atribui ao novo exemplo a classe mais comum entre esses k vizinhos.  

Este trabalho desenvolve um código de Python "from scratch" e está dividido nas seguintes partes:

1. Código de inicialização na classe KNN

Com a função __init__ são definidos 3 parâmetros: 
  - k: Número de vizinhos mais próximos a serem considerados pelo algoritmo k-NN (o padrão é 3).
  - normalize: Um bool que indica se os dados devem ser normalizados ou não (o padrão é False).
  - Atributos: O método __init__ inicializa alguns atributos da instância da classe com valores None.

2. Função fit
     É um método que é comumente usado em algoritmos de aprendizado de máquina para treinar o modelo com base nos dados de treinamento fornecidos. No contexto do k-NN (k-Nearest Neighbors), o método fit é responsável por armazenar os dados de treinamento na instância do objeto para que possam ser usados posteriormente para fazer previsões.
    Onde:
    - x_train: É o conjunto de dados de treinamento, contendo as características (features) dos exemplos de treinamento. Cada linha representa um exemplo, e cada coluna representa uma característica.
    - y_train: São os rótulos correspondentes aos exemplos de treinamento em X_train. Cada rótulo é associado a um exemplo específico e indica a classe ou o valor que o modelo deve aprender.

Dentro do método fit, os dados de treinamento (X_train e y_train) são simplesmente armazenados nos atributos da instância self.X_train e self.y_train. Isso significa que, após chamar o método fit, a instância da classe terá acesso aos dados de treinamento, e o modelo poderá ser treinado com base nesses dados.

3. Método Predict
    É responsável por fazer previsões com base nos dados de teste fornecidos. 
    - X_test: São os dados de teste nos quais serão feitas as previsões.

    Código utilizado contém:
    - Normalização (se necessário): Se o atributo normalize for True, os dados de teste (X_test) são normalizados usando a função normalize_data antes de fazer previsões. Isso garante que os dados de teste sejam processados da mesma maneira que os dados de treinamento.

    - Cálculo das Distâncias: Para cada exemplo em X_test, a distância é calculada em relação a todos os exemplos de treinamento em X_train. A função np.linalg.norm é usada para calcular a distância euclidiana.

    - Identificação dos Vizinhos Mais Próximos: Os índices dos k vizinhos mais próximos são obtidos usando argsort para ordenar as distâncias e selecionar os k primeiros índices.

    - Contagem dos Rótulos: Conta-se a ocorrência de cada rótulo nos k vizinhos mais próximos.

    - Atribuição do Rótulo Predito: O rótulo predito é determinado pelo rótulo mais comum entre os k vizinhos mais próximos.

    - Armazenamento das Previsões: As previsões são armazenadas na lista predictions.

    - Retorno das Previsões: As previsões são retornadas como um array NumPy.

4. Método normalize_data
    É utilizado para normalizar os dados, se a opção de normalização estiver ativada (self.normalize == True). Normalização é um processo comum em algoritmos de Machine Learning, onde os valores das características (features) são ajustados para garantir que estejam na mesma escala.

    Normalização dos dados incluí:
    - Cálculo dos Mínimos e Máximos: Para cada coluna (característica) em data, np.min e np.max são usados para calcular os valores mínimos e máximos.

    - Normalização: Cada valor em data é normalizado subtraindo o valor mínimo da coluna correspondente e dividindo pelo intervalo (diferença entre o máximo e o mínimo).

    - Retorno dos Dados Normalizados: Os dados normalizados são retornados pela função.

Esse método é utilizado no método predict antes de fazer previsões, caso a opção de normalização (self.normalize) esteja ativada.


