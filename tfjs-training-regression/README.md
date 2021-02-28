# TensorFlow.js 27/02/2021

Seguindo o treinamento:

[TensorFlow.js - Making Predictions from 2D Data | Google Codelabs](https://codelabs.developers.google.com/codelabs/tfjs-training-regression)

`Modelo de regressão` é aquele cuja saída é contínua, tipicamente representada por números em pontos flutuantes. O contrário do modelo de regressão é o `modelo de classificação`, cuja saída é discreta (cada classificação é discreta).

`Aprendizagem supervisionada` ocorre quando damos ao modelo em treinamento as entradas, e as saídas esperadas.

A `arquitetura do modelo` é um jeito de descrever quais funções o modelo vai executar. No caso das redes neurais, a arquitetura é a quantidade de camadas, e qual algoritmo vai ser executado em cada camada, e de que jeito.

`Inputs` são as entradas de treinamento em machine learning. `Labels` são os resultados reais que correspondem aos inputs.

`Batch` é o nome dado para cada subset entregue para treino na rede neural. `BatchSize` é o tamanho do lote em uma seção de treinamento.

`Epoch` se refere a quantidade de vezes que a rede vai olhar para o modelo de treinamento antes de finalizar. Aumentamos ou diminuimos esse número conforme vemos o comportamento de plateau do gráfico.

`Perda` ou `loss` é a medida de quão bem o algorítmo está se aproximando de um modelo eficiente a cada iteração do batch. Uma perda `Mean Squared Error` pega o quadrado da diferença do valor encontrado e o valor real e divide pelo número de exemplos.

Porquê aumentar o número de camadas ocultas faz com que a curva de valores previstos passe de primeira para segunda ordem?

## Best practices

1. O que é um tensor?
    1. Tensor é uma estrutura de dados otimizada para machine learning no TensorFlow.
    2. Podemos executar o seguinte código para criar um Tensor 2D, sendo que esse tem o formato `[num_examples, num_features_per_example]`. No caso abaixo a array tem o número de exemplos, e cada elemento da array tem 1 atributo (aqui chamado de feature).

    ```javascript
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    ```

2. O que é normalização?
    1. Normalizar os dados em um range númerico, como por exemplo, de 0 a 1.
    2. Normalizar faz com que as grandezas das features do problema não interfiram no modelo, ou no resultado. O que importa aqui é a interação entre variáveis.
3. O que é shuffling?
    1. Randomização da ordem dos elementos na array usada para treinamento. Assim ajudamos o modelo a não depender da ordem dos elementos, nem ficar enviesado pelos primeiros elementos no treinamento.