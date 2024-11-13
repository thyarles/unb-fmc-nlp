# Arquivos vinculados
# ==> exercicio2.md
# ==> exercicio2.ipynb

# Importação de pacotes para a solução do problema
import os  # Para uso do sistema operacional
import json  # Para ler jsons (na pasta corpus)
import nltk  # Natural Language Tool Kit
import glob  # Para entender padrão de arquivos
import regex as re  # Para limpar texto (regex)
import unicodedata  # Para remover caracteres não UTF-8
import random  # Para randomizar dados treino/teste
import time  # Para metir o tempo de treinamento
import math  # Para calcular perplexidade

# Importação de módulos de pacotes
from multiprocessing import cpu_count  # Para processamento paralelo
from collections import defaultdict  # Para uso no bigrama
from sklearn.model_selection import train_test_split  # Para divisão de treino/teste
from tqdm.contrib.concurrent import process_map  # Para paralelizar


class ModeloBigramas:
    def __init__(self):
        self.modelo = None
        self.sentencas = []
        self.treino = []
        self.teste = []

    def verificar_tokenizador(self):
        """Verifica ou baixa o tokenizador para português."""
        if not any(
            os.path.exists(os.path.join(caminho, "tokenizers/punkt/portuguese.pickle"))
            for caminho in nltk.data.path
        ):
            print("==> Carregando tokenizadores do NTLK...")
            nltk.download("punkt")

    def limpar_texto(self, texto):
        """Realiza pré-processamento no texto."""
        pontuacao = re.escape("!\"#%'()*+,./:;<=>?@[\\]^_`{|}~")
        regex = [
            ["", r"\{.*\}"],
            [" ", r"<(\/|\\)?.+?>"],
            [" ", r"(http|https)://[^\s]+"],
            [r'\1"', r"(?u)(^|\W)[‘’′`']"],
            [r'"\1', r"(?u)[‘’`′'](\W|$)"],
            ['"', r"(?u)[‘’`′“”]"],
            ['"', r"(\"\")"],
            ['"', r"(\'\')"],
            [".", r"(?<!\.)\.\.(?!\.)"],
            [".", r"…"],
            [" - ", r" -(?=[^\W\d_])"],
            [" - ", r"–"],
            [" ", r"[/\\|\[\]\{\}`]"],
            [" . ", r"=="],
            [" ", r"="],
            [" ", r"Categoria:\w+"],
            [" ", r"\*"],
            [r"\1", r'([,";:.]){,5}'],
            [" ", r" +"],
            [r"\1", r'\s+([,";:.])'],
            ["", r"\.(\s\w+\s?[-,:]?\s?){1,4}\."],
        ]
        texto_limpo = unicodedata.normalize("NFKD", texto)
        texto_limpo = texto_limpo.encode("utf-8", errors="ignore").decode(
            "utf-8", errors="ignore"
        )
        for regra in regex:
            texto_limpo = re.sub(pattern=regra[1], repl=regra[0], string=texto_limpo)
        return texto_limpo.strip()

    def sentenizar(self, arquivo):
        """Lê um arquivo JSON, limpa e tokeniza suas sentenças."""
        self.verificar_tokenizador()
        caminho = f"{os.getcwd()}/{arquivo}"
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = json.load(f)
                texto = conteudo.get("text", "")
                if not texto:
                    return []
                texto = self.limpar_texto(texto)
                sentencas = nltk.sent_tokenize(texto, language="portuguese")
                return [sentenca for sentenca in sentencas if len(sentenca) >= 25]
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
            return []

    def paralelizar(self, padrao, funcao, tarefas=1):
        """Processa arquivos JSON de forma paralela."""
        tarefas = cpu_count() - 1 if tarefas == "max" else tarefas
        arquivos = glob.glob(padrao)
        print(f"==> Processando {len(arquivos)} JSONs...")
        inicio = time.time()
        resultados = process_map(funcao, arquivos, max_workers=tarefas, chunksize=1)
        resultado_final = [res for sub in resultados for res in sub]
        fim = time.time()
        print(f"    feito em {(fim - inicio):.2f} segundos.")
        self.sentencas = resultado_final

    def separar_teste(self, percentual_teste=0.2):
        print(f"==> Separando {len(self.sentencas)} sentenças em treino e teste...")
        treino, teste = train_test_split(
            self.sentencas, test_size=float(percentual_teste)
        )
        print(f"    Treino: {len(treino)} sentenças, Teste: {len(teste)} sentenças.")
        self.treino = treino
        self.teste = teste

    def bigramas(self):
        """Calcula probabilidades dos bigramas."""
        print("==> Calculando probabilidades dos bigramas...")
        inicio = time.time()
        bigramas = defaultdict(lambda: defaultdict(int))
        unigramas = defaultdict(int)

        for sentenca in self.sentencas:
            palavras = ["<s>"] + sentenca.split() + ["</s>"]
            for i in range(len(palavras) - 1):
                unigramas[palavras[i]] += 1
                bigramas[palavras[i]][palavras[i + 1]] += 1
            unigramas[palavras[-1]] += 1

        probabilidades = {
            palavra: {
                proxima: count / unigramas[palavra]
                for proxima, count in seguinte.items()
            }
            for palavra, seguinte in bigramas.items()
        }
        fim = time.time()
        print(f"    feito em {(fim - inicio):.2f} segundos.")
        self.modelo = probabilidades

    def gerar_texto(self, raiz=None, minimo=100, maximo=2000):
        """Gera texto com base no modelo de bigramas."""
        if not self.modelo:
            raise ValueError("O modelo não foi treinado.")
        inicio, fim = "<s>", "</s>"
        texto = [] if raiz is None else raiz.split()
        palavra_atual = inicio if raiz is None else texto[-1]

        for _ in range(maximo - 1):
            palavra_proxima = self.modelo.get(palavra_atual, None)
            if not palavra_proxima:
                texto.append("_______.")
                palavra_atual = inicio
                continue
            palavra_atual = random.choices(
                population=list(palavra_proxima.keys()),
                weights=list(palavra_proxima.values()),
            )[0]
            if palavra_atual == fim and len(texto) >= minimo:
                break
            if palavra_atual not in [inicio, fim]:
                texto.append(palavra_atual)
        texto = re.sub(r"\s+([.,!?])", r"\1", " ".join(texto))
        return " ".join(nltk.sent_tokenize(texto, language="portuguese")).strip()

    def perplexidade(self):
        """Calcula a perplexidade do modelo."""
        print("==> Calculando a perplexidade...")
        logaritmo_probabilidade = 0
        quantidade_palavras = 0

        if not self.modelo:
            raise ValueError("O modelo não foi treinado.")

        for sentenca in self.teste:
            palavras = sentenca.split()
            for i in range(len(palavras) - 1):
                palavra, proxima_palavra = palavras[i], palavras[i + 1]
                probabilidade = self.modelo.get(palavra, {}).get(proxima_palavra, 1e-8)
                logaritmo_probabilidade += math.log(probabilidade)
                quantidade_palavras += 1

        media = logaritmo_probabilidade / quantidade_palavras
        valor_perplexidade = math.exp(-media)
        print(f"    perplexidade = {valor_perplexidade:.2f}.")

    def processar_frase(self, frase):
        """Processa frasses de forma paralela."""
        a, e = 0, 0
        palavras = frase.split()
        for i in range(len(palavras) - 1):
            try:
                if (
                    self.gerar_texto(raiz=palavras[i], minimo=1, maximo=2).split()[-1]
                    == palavras[i + 1]
                ):
                    a += 1
                else:
                    e += 1
            except IndexError:
                pass  # Exceção esperada ao acessar palavras além do limite
        return a, e

    def testar_modelo(self, tarefas=1):
        """Testa o modelo usando um conjunto de sentenças de teste."""
        if not self.modelo:
            raise ValueError("O modelo não foi treinado.")

        print("==> Testando o modelo...")
        inicio = time.time()
        tarefas = cpu_count() - 1 if tarefas == "max" else tarefas
        acertos, erros = 0, 0

        # Processa as frases de teste em paralelo
        resultados = process_map(
            self.processar_frase, self.teste, max_workers=tarefas, chunksize=500
        )

        # Agregando os resultados
        for a, e in resultados:
            acertos += a
            erros += e

        fim = time.time()
        print(f"    feito em {(fim - inicio):.2f} segundos.")
        print(f"    Acertos: {acertos}, Erros: {erros}")
        print(f"    Percentual de acerto: {acertos/(erros+acertos)}")


# Exemplo de uso
if __name__ == "__main__":
    modelo = ModeloBigramas()
    modelo.paralelizar("corpus_test/*.json", modelo.sentenizar, "max")
    modelo.separar_teste()
    modelo.bigramas()
    modelo.perplexidade()
    print(modelo.gerar_texto())
    print(modelo.gerar_texto(raiz="O Charles é"))
    modelo.testar_modelo(tarefas="max")
