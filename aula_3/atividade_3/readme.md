# Greedy Search

Solução para a [atividade 3](https://github.com/thyarles/unb-fmc-nlp/blob/main/aula_3/readme.md), estruturada da seguinte forma:

## Bibliotecas

* [`scikit-learn`](https://scikit-learn.org/0.21/documentation.html)
* [`pandas`](https://pandas.pydata.org/docs/) (versões no arquivo [`requirements.txt`](https://github.com/thyarles/unb-fmc-nlp/blob/main/aula_3/atividade_3/requirements.txt)).

## Modelos implementados

### Multinomial Naive Bayes

O Multimonial Naive Bayes é um modelo classificador probabilístico baseado no teorema de Bayes, adequado para classificação de documentos textuais. A probabilidade calculada é a de um documento pertencer a determinada classe com base na ocorrência de palavras independentes.

Este modelo tem como principais parâmetros:

* `alpha`

    Suavização (evita probabilidades nulas).
* `fit_prior`

    Se aprende com probabilidades anteriores.

### Regressão Logística

A Regressão Logística é um modelo estatístico que utiliza função logística para modelar uma variável dependente binária, estimando se determinado texto  pertencer ou não a uma determinada classe.

Este modelo tem como principais parâmetros:

* `C`
    
    Força da regularização (para evitar overfitting).

* `solver`

    Algoritmos possíveis de otimização (`lbfg`, `liblinear` e `sag`).

* `penalty`

    Tipos de regularização (`l1`, `l2`, `elasticnet` e `None`).

* `max_iter`

    Número máximo de iterações.

### SVM (_Support Vector Machine_)

O SVM é um modelo que analisa o texto de entrada para encontrar o melhor hiperplano (vetor que maximiza a distância entre os pontos de diferentes classes).

Este modelo tem como principais parâmetros:

* `C`

    Força da regularização.

* `loss`

    Função de perda (`hinge` ou `squared_hinge`).

* `max_iter`

    Número máximo de iterações.

* `tol`

    Critério de tolerância de parada.

## Resultados

* Para os testes dos algoritmos, utilizou-se as bases [`CSTR`](https://github.com/ragero/text-collections/blob/master/complete_texts_csvs/CSTR.csv) e [`Dmoz-Computers`](https://github.com/ragero/text-collections/blob/master/complete_texts_csvs/Dmoz-Computers.csv).

* No `Greedy Search`, ignorou-se resultados com `score` superiores a `0.997` para evitar um possível over-fitting. Ainda, combinações de parâmetros impossíveis foram igualmente ignoradas.

A seguir, uma análise dos resultados apenas para a base `CSTR`, dado que os resultados foram semelhantes para as duas bases.

### Base CSTR

#### Multinomial Naive Bayes

* 5 Melhores Resultados

    | Index | Alpha | Fit Prior | Score          |
    |-------|-------|-----------|----------------|
    | 0     | 0.1   | True      | 0.974895397490 |
    | 1     | 0.1   | False     | 0.974895397490 |
    | 2     | 0.25  | True      | 0.974895397490 |
    | 3     | 0.25  | False     | 0.974895397490 |
    | 4     | 0.5   | True      | 0.974895397490 |

    Todos os melhores resultados (score `0.9749`) são obtidos com valores de `alpha` abaixo de `2.0`.

    A variável `fit_prior` não impacta diretamente os melhores scores, pois tanto `True` quanto `False` aparecem nos primeiros lugares.


* 5 Piores Resultados

    | Index | Alpha | Fit Prior | Score          |
    |-------|-------|-----------|----------------|
    | 10    | 2.0   | True      | 0.970711297071 |
    | 12    | 5.0   | True      | 0.866108786611 |
    | 13    | 5.0   | False     | 0.866108786611 |
    | 11    | 2.0   | False     | 0.974895397490 |*

    > Nota: O resultado com índice 11 foi incluído porque, apesar de ter um valor de alpha menor que 5.0, o score foi igual aos melhores, o que impacta a análise.

    O desempenho cai significativamente quando o valor de `alpha` chega a `5.0`. Ambos os valores de `fit_prior` (`True` e `False`) produzem os piores scores (`0.8661`).

    O valor de `alpha = 2.0` (com `fit_prior = True`) apresentou uma leve queda no desempenho (`0.9707`) em relação aos melhores, sugerindo que valores intermediários de `alpha` começam a degradar o desempenho.

 #### Regressão Logística

 * 5 Melhores Resultados

    | Index | C    | Solver   | Penalty | Max Iter | Score          |
    |-------|------|----------|---------|----------|----------------|
    | 0     | 0.1  | lbfgs    | l2      | 100      | 0.974895397490 |
    | 1     | 0.1  | lbfgs    | l2      | 200      | 0.974895397490 |
    | 2     | 0.1  | lbfgs    | l2      | 500      | 0.974895397490 |
    | 3     | 0.1  | lbfgs    | l2      | 1000     | 0.974895397490 |
    | 4     | 0.1  | lbfgs    |         | 100      | 0.974895397490 |

    Todos os melhores resultados (score `0.9749`) são obtidos com valores de `C = 0.1` e diferentes configurações de `max_iter`.  
    
    O parâmetro `penalty` pode ser omitido ou configurado como `l2`, sem impacto no score.  

    O solver `lbfgs` aparece em todos os melhores resultados.


* 5 Piores Resultados

    | Index | C    | Solver   | Penalty | Max Iter | Score          |
    |-------|------|----------|---------|----------|----------------|
    | 8     | 0.1  | liblinear| l1      | 100      | 0.778242677824 |
    | 9     | 0.1  | liblinear| l1      | 200      | 0.778242677824 |
    | 10    | 0.1  | liblinear| l1      | 500      | 0.778242677824 |
    | 11    | 0.1  | liblinear| l1      | 1000     | 0.778242677824 |
    | 32    | 0.25 | liblinear| l1      | 100      | 0.920502092050 |

    O desempenho mais baixo (score `0.7782`) é obtido com o solver `liblinear` configurado para a penalidade `l1` e `C = 0.1`.  
    Uma penalidade `l1` continua a impactar negativamente os resultados mesmo com valores de `C` um pouco maiores, como `0.25` (score `0.9205`).  

    O parâmetro `max_iter` não altera significativamente o desempenho para estas combinações específicas. O solver `liblinear` com a penalidade `l1` tem um impacto negativo claro no desempenho, enquanto o solver `lbfgs` com `l2` oferece os melhores resultados consistentes.  

    Recomenda-se evitar `penalty = l1` ao usar o solver `liblinear`, especialmente para valores baixos de `C`.

#### SVM

TODO.

    