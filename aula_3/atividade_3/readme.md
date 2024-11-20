# Greedy Search

Solução para a [atividade 3](https://github.com/thyarles/unb-fmc-nlp/blob/main/aula_3/readme.md), estruturada da seguinte forma:

## Bibliotecas

* [`scikit-learn`](https://scikit-learn.org/0.21/documentation.html)
* [`pandas`](https://pandas.pydata.org/docs/) (versões no arquivo [`requirements.txt`](https://github.com/thyarles/unb-fmc-nlp/blob/main/aula_3/atividade_3/requirements.txt)).

## Modelos implementados

### Multinomial Naive Bayes

O Multimonial Naive Bayes é um modelo classificador probabilístico baseado no teorema de Bayes, adequado para classificação de documentos textuais. A probabilidade calculada é a de um documento pertencer a determinada classe com base na ocorrência de palavras independentes.

Este modelo tem como principais parâmetros:

* Suavização (evita probabilidades nulas).
* Número de classes.

### Regressão Logística

A Regressão Logística é um modelo estatístico que utiliza função logística para modelar uma variável dependente binária, estimando se determinado texto  pertencer ou não a uma determinada classe.

Este modelo tem como principais parâmetros:

* Regularização (para evitar overfitting).
* Limiar de decisão (nível de separação das classes).

### SVM (_Support Vector Machine_)

O SVM é um modelo que analisa o texto de entrada para encontrar o melhor hiperplano (vetor que maximiza a distância entre os pontos de diferentes classes).

Este modelo tem como principais parâmetros:

* Kernel (mapeamento de dados).
* C (controle da marge e erro).
