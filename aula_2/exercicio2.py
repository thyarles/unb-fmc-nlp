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
      caminho_nltk = os.path.join(caminho, "tokenizers/punkt/portuguese.pickle")
      if os.path.exists(caminho_nltk):
        baixado = True
        break

  # Baixa se não encontrou
  if not baixado:
    print("Carregando tokenizadores do NTLK...")
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        print("Baixado com sucesso!")
    except Exception as e:
        print(f"Erro ao baixar o tokenizador: {e}")


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
    [' ',     r'[/\|]'],                          # Barras
    [' . ',   r'=='],                             # Títulos
    [' ',     r'Categoria:\w+'],                  # Tags
    ['. ',    r' \* '],                           # Asteriscos
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
  texto_limpo = unicodedata.normalize("NFKD", texto)
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
  try:
    with open(caminho, 'r', encoding='utf-8') as f:
      conteudo = json.load(f)
      texto = conteudo.get("text", "")
      if not texto:
        return []
      texto = limpar(texto)
      sentencas = nltk.sent_tokenize(texto, language='portuguese')
      # Retorna só os maiores que 25 caracteres (o regex não está 100%)
      return [sentenca for sentenca in sentencas if len(sentenca) >= 25]
  except Exception as e:
    print(f"Erro ao processar {arquivo}: {e}")
    return []


# Função para carregar JSONs de forma paralela
def paralelizar(padrao, funcao, tarefas=1):
  # Se max, vamos usar todas as CPUs
  tarefas = cpu_count()-1 if tarefas=='max' else tarefas
  # Monta caminho
  arquivos = glob.glob(padrao)
  # Processando
  with Pool(tarefas) as p:
      parcial = p.map(funcao, arquivos)
  # Transforma em lista
  result = [res for sub in parcial for res in sub]
  return result


# Função para calcular bigramas
def calcular_bigramas(sentencas):
  # Prepara dicionários com auto inicialização
  bigramas  = defaultdict(lambda: defaultdict(int))
  unigramas = defaultdict(int)

  for sentenca in sentencas:
    palavras = nltk.word_tokenize(sentenca, language='portuguese')
    # Adiciona marcação de sentença
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

  return probabilidades   


# Testes temporários
if __name__ == "__main__":
  sentencas = paralelizar('corpus_test/*.json', sentenizar, 'max')
  treino, teste = train_test_split(sentencas, test_size=0.2)
  bigramas = calcular_bigramas(treino)
  print("fim")