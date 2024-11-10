# Arquivos vinculados
# ==> exercicio2.md
# ==> exercicio2.ipynb (TODO)

# Importação de pacotes para a solução do problema
import os           # Para uso do sistema operacional
import json         # Para ler jsons (na pasta corpus)
import nltk         # Natural Language Tool Kit


# Função para ler JSON no disco e tokenizar
def tokenizar_json(arquivo):
  caminho = f'{os.getcwd()}/{arquivo}'
  try:
    with open(caminho, 'r', encoding='utf-8') as f:
      conteudo = json.load(f)
      texto = conteudo.get("text", "")
      # Instala pacote para língua portuguesa
      nltk.download('punkt_portuguese')
      tokens = nltk.sent_tokenize(texto)
      sentencas = [f"<s> {sentenca} </s>" for sentenca in tokens]
      return sentencas
  except Exception as e:
    print(f"Erro ao processar {arquivo}: {e}")
    return []

# Testes temporários
tokenizar_json('corpus/240.json')