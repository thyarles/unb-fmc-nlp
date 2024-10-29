# Atividade: Criação de Modelo de Bigrama para Geração de Texto

## Objetivo

Nesta atividade, você irá construir um modelo de bigrama para geração de texto. A atividade será realizada utilizando o tokenizador que você implementou na atividade anterior (algoritmo BPE). O objetivo é treinar e avaliar o modelo, bem como medir sua perplexidade em um conjunto de dados de teste.

## Instruções

1. **Preparação dos Dados**:
   - Utilize o tokenizador implementado na atividade anterior para segmentar o conjunto de dados fornecido.
   - Divida os arquivos em **treino** (80%) e **teste** (20%) de forma aleatória.
  
2. **Implementação do Modelo de Bigrama**:
   - Implemente um modelo de bigrama em Python. O modelo deve calcular a probabilidade condicional de uma palavra dado a palavra anterior com base nos dados de treino.
   - A saída deve ser o modelo que calcula a distribuição de probabilidade das palavras baseando-se nas palavras anteriores.
   
3. **Cálculo da Perplexidade**:
   - Aplique o modelo de bigrama no conjunto de dados de teste para calcular a **perplexidade**, uma métrica usada para avaliar a capacidade preditiva do modelo. A perplexidade indica o quão bem o modelo prevê o próximo termo em uma sequência.

4. **Geração de Texto**:
   - Implemente uma função que gera texto a partir do modelo de bigrama.
   - No *notebook*, gere um exemplo de texto com pelo menos 20 tokens para demonstrar o funcionamento do modelo.

5. **Entrega**:
   - Crie um arquivo `.py` com a implementação do modelo bigrama e a função de cálculo de perplexidade. Certifique-se de que o código está funcional e bem documentado.
   - No *notebook*, carregue o modelo e o conjunto de dados de teste, e aplique as seguintes funções:
     - Exemplo de geração de texto a partir do modelo de bigrama.
     - Apresentação do cálculo da perplexidade para o conjunto de teste.

## Estrutura Esperada

1. **Arquivo Python (.py)**:
   - Código do modelo de bigrama.
   - Função de cálculo de perplexidade.
   - Função de geração de texto.

2. **Notebook (.ipynb)**:
   - Apresentação do exemplo de geração de texto.
   - Cálculo e exibição da perplexidade do modelo no conjunto de teste.
  

## Dicas

- Utilize a fórmula da perplexidade indicada no livro texto da disciplina -- [Link do Livro](https://web.stanford.edu/~jurafsky/slp3/).
  
- No vídeo do Andrej Karpathy ([link do vídeo](https://www.youtube.com/watch?v=PaCmpygFfXo)) ele descreve e implementa um modelo bigrama. Aproveitem a ótima aula dele usem algumas facilidades como a função multinomial para amostragem. **A partir de 1h desse vídeo ele implementa usando Redes Neurais, ignorem essa parte!** (trabalho futuro?)
  
- Para a **geração de texto**, experimente iniciar com uma palavra ou token aleatório do vocabulário e gere a sequência a partir do modelo até atingir o número desejado de tokens.

## Avaliação

- Implementação correta do modelo de bigrama e da função de perplexidade.
- Organização e clareza do código.
- Documentação e comentários explicativos.
- Correta aplicação do tokenizador desenvolvido anteriormente.

**Prazo**: Submeta a atividade até **05/11/2024** na plataforma da disciplina (plataforma a definir).
