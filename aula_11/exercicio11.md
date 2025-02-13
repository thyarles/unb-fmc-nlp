# Orientação de Exercícios de PLN

## Objetivo

Nas últimas duas aulas foram descritas a implementação do GPT. Foram apresentadas a implementação do mecanismo de Atenção e todas as camadas da arquitetura Transformer. Porém, ainda restou a parte de pré-treinamento. Assim, para cobrir essa última e importante parte, o exercícios corresponde ao Capítulo 5 do livro do Sebastian Raschka "Build a Large Language Model From Scratch".
Acompanhe o conteúdo no notebook abertamente disponível no github:  [Capitulo 5](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb).


#### As duas bases de dados serão as mesmas escolhidas e descritas na Atividade 3 (Veja a planilha [Aqui](https://docs.google.com/spreadsheets/d/1F485czBA5zR60J4efsEz8-4YydywazOPCKVjb1o0jUs/edit?gid=0#gid=0) ). 

Data da entrega: **04/02/2024**

## Classificadores e Abordagem

1. **Modelos a implementar**:
   - BERT (qualquer modelo da família BERT que você seja capaz de executar com seus recursos computacionais)

3. **Divisão da base**:
   - Divida a base em treino (70%), validação (10%) e teste (20%).   

---

### Crie um **notebook** que:
   - Carregue a base de dados e faça a divisão entre treino, validação e teste.
   - Treine o BERT (antes, faça a tokenização e veja como estão os tokens de um documento!) 
   - Apresente os resultados de F1-score (micro e macro), acurácia e a matriz de confusão.

---

## Entregáveis

1. **Notebook**:
   - Apresente:
     - Os resultados (f1-score, acurácia e matriz de confusão)
     - As métricas finais calculadas no conjunto de teste.

---
