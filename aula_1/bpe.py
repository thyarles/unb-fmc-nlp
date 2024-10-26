# BPE (Byte-Pair Encoding)

# Quanto maiores forem as modificações, maior será o vocabulário e menor será
# o texto. Precisamos encontrar o equilíbrio.

# __init__      : chama o criador do vocabulario e um dicionário vazio
# _build_vocab  : cria o vocabulário
# _get_stats    : conta a ocorrência dos pares em determinada string
# _merge        : troca as ocorrências de maior frequência por novo vocabulário
# train         : procura os caracteres frequentes e une, criando novo vocabulário
# encoding      : converte texto em ids
# decoding      : converte ids em texto
# print_subwords: imprime a representação criada no treinamento

class MyBPE():
    
    # Inicializa dicionário para merges e vocabulário, tornando-os disponíveis
    # durante toda existência da classe
    def __init__(self):
        self.merges = {} 
        self.vocab = self._build_vocab()

    # Cria o vocabulário, é chamada imediatamente após a instância da classe
    def _build_vocab(self):
        # Cria o vocabulário inicial, com valores de 0 a 255
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Estende o vocabulário com base nos valores que tiveram merge
        # Observe que a soma é uma concatenação, pois vocab é byte
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab


    # Conta as ocorrências de cada par em determinado ids
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            # Incrementa a cada par
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    # Outra forma de fazer (versão mais humana e menos pytônica)
      # def _get_stats(ids):
      # counts = {}
      # for i in range(len(ids) - 1):
      #     pair = (ids[i], ids[i + 1])
      #     if pair in counts:
      #         counts[pair] += 1
      #     else:
      #         counts[pair] = 1
      # return counts


    # Troca determinado par por um novo vocabulário
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            # Verifica o par na sequência e pula se houver merge (+2)
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            # Não teve merge, mantenha o valor e vai para o próximo
            else:
                newids.append(ids[i])
                i += 1
        return newids


    # Treina o modelo adicionando cada par unido no dicionário
    def train(self, text, vocab_size):
        # Se o vocabulário for maior que 255, sinalize o erro
        assert vocab_size >= 256
        # O vocabulário já tem 256, calcule a diferença
        num_merges = vocab_size - 256
        # Converta o texto de entrada para bytes, usando unicode UTF-8
        text_bytes = text.encode("utf-8")
        # Cada byte tem que ser um elemento independente
        ids = list(text_bytes)
        # Dicionário para salvar o processamento local
        merges = {}
        # Vocabulário com os valores de 0 a 255
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # Pega a frequênica dos pares
            stats = self._get_stats(ids)
            # Procura pelo de maior frequência
            pair = max(stats, key=stats.get)
            # Incrementa identificador do vocabulário
            idx = 256 + i
            # Faz a união do par no identificador idx
            ids = self._merge(ids, pair, idx)
            # Registra como uma união para operações de [de]codificação
            merges[pair] = idx
            # Atualiza vocabulário
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        # Salva os merges na instância
        self.merges = merges
        # Salva o vocabulário na instância
        self.vocab = vocab


    # Decodifica determinados ids em texto
    def decode(self, ids):
        # Cria a cadeia de bytes
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Decodifica usando UTF-8: se tiver erro de conversão, troque
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    # Codifica determinado texto em ids
    def encode(self, text):
        # Converte o texto para byte usando UTF-8
        text_bytes = text.encode("utf-8")
        # Converte em lista para que cada byte seja independente
        ids = list(text_bytes)
        # A lista tem que ter pelo menos 2 elementos, do contrário
        # não faz qualquer sentido o merge
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # float('inf') é um fallback, um número gigante que garantirá um
            # valor mínimo caso tenha algo errado no index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                # Se chegou aqui, não há nada mais para unir, saia
                break
            # Recupere o idx da instância
            idx = self.merges[pair]
            # Recuere o ids da instância e faça a união do par
            ids = self._merge(ids, pair, idx)
        return ids


    # Imprime a lista de subpalavras criadas no treinamento
    def print_subwords(self):
      merges = sorted(self.merges)
      # Como o vocabulário base tem 256 itens, vamos começar do 257
      i = 257
      for tokens in merges:
        print(f'Vocabulary {i} with tokens {tokens} decodes to "{self.decode(tokens)}"')
        i += 1
