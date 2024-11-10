# Arquivos vinculados
# ==> exercicio2.md
# ==> exercicio2.ipynb (TODO)

# Importação de pacotes para a solução do problema
import os           # Para uso do sistema operacional
import json         # Para ler jsons (na pasta corpus)
import nltk         # Natural Language Tool Kit
import glob         # Para entender padrão de arquivos
import regex as re  # Para limpar texto (regex)
import unicodedata  # Para remover caracteres não UTF-8
import random       # Para randomizar dados treino/teste
import time         # Para metir o tempo de treinamento
import math         # Para calcular perplexidade

# Importação de módulos de pacotes
from multiprocessing import Pool, cpu_count           # Para processamento paralelo
from collections import defaultdict                   # Para uso no bigrama
from sklearn.model_selection import train_test_split  # Para dividir dados


# Função para verificar dados do tokenizador
def verificar():
  # Verifica pacote Pickle para língua portuguesa do NLTK
  baixado = False

  # Testa todos os caminhos possíveis
  for caminho in nltk.data.path:
      caminho_nltk = os.path.join(caminho, 'tokenizers/punkt/portuguese.pickle')
      if os.path.exists(caminho_nltk):
        baixado = True
        break

  # Baixa se não encontrou
  if not baixado:
    print('==> Carregando tokenizadores do NTLK...')
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        print('==> Baixado com sucesso!')
    except Exception as e:
        print(f'Erro ao baixar o tokenizador: {e}')


# Função para limpar texto (pré-processamento)
def limpar(texto):
  # Inspirado em https://github.com/eberlitz/pt-br-corpus/blob/master/scripts/preprocess.py
  pontuacao = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')
 
  regex = [
    ['',      r'\{.*\}'],                         # Brakets
    [' ',     r'<(\/|\\)?.+?>'],                  # HTML
    [' ',     r'(http|https)://[^\s]+'],          # URL,
    [r'\1"',  r"(?u)(^|\W)[‘’′`']"],              # Normaliza quotas
    [r'"\1',  r"(?u)[‘’`′'](\W|$)"],              # Normaliza quotas
    ['"',     r'(?u)[‘’`′“”]'],                   # Normaliza quotas
    ['"',     r'(\"\")'],                         # Normaliza quotas
    ['"',     r'(\'\')'],                         # Normaliza quotas
    ['.',     r'(?<!\.)\.\.(?!\.)'],              # Normaliza ...
    ['.',     r'…'],                              # Normaliza ...
    [' - ',   r' -(?=[^\W\d_])'],                 # Hífens
    [' - ',   r'–'],                              # Hífens
    [' ',     r'[/\\|\[\]\{\}`]'],                # Barras e sinais
    [' . ',   r'=='],                             # Títulos
    [' ',     r'='],                              # Qualquer igualdade
    [' ',     r'Categoria:\w+'],                  # Tags
    [' ',     r'\*'],                             # Asteriscos e iguais
    [r'\1',   r'([,";:.]){,5}'],                  # Pontuação duplicada
    [' ',     r' +'],                             # Espaços dobrados
    [r'\1',   r'\s+([,";:.])'],                   # Pontuação com espaço
    ['',      r'\.(\s\w+\s?[-,:]?\s?){1,4}\.']    # Frases com poucas palavras
    # Ajustes de pontuações
    # [r'\1 \2 \3', r'(\w+)([%s])([ %s])' % (pontuacao, pontuacao)], 
    # [r'\1 \2 \3', r'([ %s])([%s])(\w+)' % (pontuacao, pontuacao)],
    # [r'\1 \2',    r'(\w+)([%s])$' % (pontuacao)]
  ]

  # Remove caracteres que não são UTF-8
  texto_limpo = unicodedata.normalize('NFKD', texto)
  texto_limpo = texto_limpo.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

  for regra in regex:
     texto_limpo = re.sub(pattern=regra[1], repl=regra[0], string=texto_limpo)
  
  return texto_limpo.strip()


# Função para ler JSON no disco e tokenizar
def sentenizar(arquivo):

  # Verifica tokenizador
  verificar()

  # Lê JSON
  caminho = f'{os.getcwd()}/{arquivo}'

  print(f'==> Processando {caminho}...')

  try:
    with open(caminho, 'r', encoding='utf-8') as f:
      conteudo = json.load(f)
      texto = conteudo.get('text', '')
      if not texto:
        return []
      texto = limpar(texto)
      sentencas = nltk.sent_tokenize(texto, language='portuguese')
      # Retorna só os maiores que 25 caracteres (o regex não está 100%)
      return [sentenca for sentenca in sentencas if len(sentenca) >= 25]
  except Exception as e:
    print(f'Erro ao processar {arquivo}: {e}')
    return []


# Função para carregar JSONs de forma paralela
def paralelizar(padrao, funcao, tarefas=1):
  # Se max, vamos usar todas as CPUs
  tarefas = cpu_count()-1 if tarefas=='max' else tarefas
  # Monta caminho
  arquivos = glob.glob(padrao)
  print(f'==> Processando {len(arquivos)} JSONs...')
  inicio = time.time()

  # Processando
  with Pool(tarefas) as p:
      parcial = p.map(funcao, arquivos)
  # Transforma em lista
  result = [res for sub in parcial for res in sub]
  fim = time.time()
  print(f'    feito em {(fim - inicio):.2f} segundos.')
  return result


# Função para calcular bigramas
def calcular_bigramas(sentencas):
  print('==> Calculando probabilidades dos bigramas...')
  inicio = time.time()

  # Prepara dicionários com auto inicialização
  bigramas  = defaultdict(lambda: defaultdict(int))
  unigramas = defaultdict(int)

  for sentenca in sentencas:
    # Com esse ficou muito quebrado
    # palavras = nltk.word_tokenize(sentenca, language='portuguese')
    # Aqui tive melhor resultados
    palavras = sentenca.split()
    # Adiciona marcação de sentença ao split
    palavras = ['<s>'] + palavras + ['</s>']
    for i in range(len(palavras) - 1):
      unigramas[palavras[i]] += 1
      bigramas[palavras[i]][palavras[i + 1]] += 1
    unigramas[palavras[-1]] += 1  # Última palavra

  # Converter contagens em probabilidades
  probabilidades = {}
  for palavra, seguinte in bigramas.items():
    probabilidades[palavra] = {
      proxima_palavra: count / unigramas[palavra]
      for proxima_palavra, count in seguinte.items()
    }

  fim = time.time()
  print(f'    feito em {(fim - inicio):.2f} segundos.')
  print(f'    probabilidade gerada com {len(probabilidades)} termos.')
  return probabilidades

# Função para gerar texto
def gerar_texto(probabilidades, raiz=None, minimo=None, maximo=None):
  # Marcações de sentença
  inicio = '<s>'
  fim = '</s>'

  # Sem mínimo, produzir pelo menos 100 palavras
  minimo = 100 if minimo is None else int(minimo)
  # Sem máximo, produzir o limite de 2000 palavras
  maximo = 2000 if maximo is None else int(maximo)
  # Sem raiz, produzir do <s>
  if raiz is None:
    texto = []
    palavra_atual = inicio  
  else:
    texto = raiz.split()
    palavra_atual = texto[-1]

  for _ in range(maximo - 1):
    # Pega próxima, se não tiver, mostre None
    palavra_proxima = probabilidades.get(palavra_atual, None)
    if not palavra_proxima:
      texto.append('_______.')
      palavra_atual = inicio
      continue
    # palavra_atual = max(palavra_proxima, key=palavra_proxima.get)
    # Pega próxima palavra com maior probabilidade
    palavra_atual = random.choices(
      population=list(palavra_proxima.keys()),
      weights=list(palavra_proxima.values())
    )[0]
    if palavra_atual == fim and len(texto) >= minimo: break
    if palavra_atual != inicio and palavra_atual != fim:
      texto.append(palavra_atual)
  texto = re.sub(r'\s+([.,!?])', r'\1', ' '.join(texto))
  texto = nltk.sent_tokenize(texto, language='portuguese')
  return ' '.join(texto).strip()

# Função para calcular a perplexidade
def perplexidade(probabilidades, sentencas):

  # Baixa perplexidade (1-100): modelo com boa predição
  # Alta perplexidade (100-1000): modelo não tem boa predição
  # Perplexidade perfeita: 1
  # Perplexidade imperfeita: infinito (modelo totalmente aleatório)

  print('==> Calculando a perplexidade...')
  logaritmo_probabilidade = 0
  quantidade_palavras = 0

  for sentenca in sentencas:
    palavras = sentenca.split()
    for i in range(len(palavras) - 1):
      palavra, proxima_palavra = palavras[i], palavras[i + 1]
      # Temos que evitar log de zero
      probabilidade = probabilidades.get(palavra, {}).get(proxima_palavra, 1e-8)
      logaritmo_probabilidade += math.log(probabilidade)
      quantidade_palavras += 1

  media = logaritmo_probabilidade / quantidade_palavras
  valor_perplexidade = math.exp(-media)
  print(f'    perplexidade = {valor_perplexidade:.2f}.')
  return valor_perplexidade


# Testes temporários
if __name__ == '__main__':
  sentencas = paralelizar('corpus/*.json', sentenizar, 'max')
  treino, teste = train_test_split(sentencas, test_size=0.2)
  probabilidades = calcular_bigramas(treino)
  perplexidade(probabilidades, sentencas)
  print(gerar_texto(probabilidades))
  print(gerar_texto(probabilidades, raiz='O Charles é'))