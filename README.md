K-Nearest Neighbors (kNN)

O KNN, ou "k-Nearest Neighbors", é um algoritmo utilizado tanto para classificação como para regressão. Ele é conhecido como um método de aprendizado baseado em instância ou método preguiçoso (lazy) porque não tenta construir um modelo explícito durante a fase de treinamento. Em vez disso, ele memoriza os exemplos de treinamento e faz previsões com base na proximidade (similaridade) dos novos exemplos aos exemplos existentes.

A ideia básica do k-NN é que os exemplos em um espaço de características semelhantes tendem a ter rótulos semelhantes. Ou seja, ao prever o rótulo de um novo exemplo, o algoritmo olha para os k exemplos de treinamento mais próximos a ele no espaço de características e atribui ao novo exemplo a classe mais comum entre esses k vizinhos.  

Este trabalho desenvolve um código de Python "from scratch" e está dividido nas seguintes partes:

1. Código de inicialização na classe KNN

Com a função __init__ são definidos 3 parâmetros: 
  - k: Número de vizinhos mais próximos a serem considerados pelo algoritmo k-NN (o padrão é 3).
  - normalize: Um bool que indica se os dados devem ser normalizados ou não (o padrão é False).
  - Atributos: O método __init__ inicializa alguns atributos da instância da classe com valores None.

2. 

