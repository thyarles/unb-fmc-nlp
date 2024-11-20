import os
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score


class AtividadeTres:
    """
    Classe para buscar hiper-parâmetros, treinar o modelo, avaliar, salvar e carregar os resultados. 
    """

    def __init__(self, random_state:int=None):
        """
        Inicializa a classe configurando randomizador para possibilitar reprodução.
        
        Args:
            random_state (int): Semente para reprodução. Valor padrão = None.
        """
        self.random_state = random_state

    @staticmethod
    def load_data(filepath:str):
        """
        Lê dados CSV.
        
        Args:
            filepath (str): Caminho para o arquivo CSV.
        
        Returns:
            pd.DataFrame: Arquivo convertido em Pandas.
        """
        return pd.read_csv(filepath)

    def split_data(self, data: pd.DataFrame, label_col: str, test_size:float=0.2):
        """
        Divide os dados em treino e teste.
        
        Args:
            data (pd.DataFrame): Dados para dividir.
            label_col (str): Coluna com a classificação.
            test_size (float): Tamanho da base de teste. Valor padrão = 0.2.
        
        Returns:
            tuple: Train-test splits (X_train, X_test, y_train, y_test).
        """
        X = data.drop(columns=[label_col])
        y = data[label_col]

        if self.random_state:
            print(f'Dados divididos com tamanho do teste {test_size:0.2f} e semente {self.random_state}!')
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        print(f'Dados divididos com tamanho do teste {test_size:0.2f} e sem semente!')
        return train_test_split(X, y, test_size=test_size)

    @staticmethod
    def save_data_frame(data:pd.DataFrame, filename:str, path:str='out/'):
        """
        Salva dados treinados para o disco.
        
        Args:
            train (pd.DataFrame): Dados de treinamento..
            test (pd.DataFrame): Dados de teste.
            file (str): Nome para o arquivo.
            path (str): Diretório para salvar. Valor padrão = 'out/'.
        """
        os.makedirs(path, exist_ok=True)
        data.to_csv(os.path.join(path, filename), index=False)
        print(f'Arquivo {path}{filename} salvo com sucesso!')

    @staticmethod
    def create_param_grid(param_dict:dict):
        """
        Monta conjunto de hiper parâmetros para uso no greedy_search.
        
        Args:
            param_dict (dict): Dicionário com parâmetros e valores.
        
        Returns:
            list: Lista de dicionário com a combinação dos hiper parâmetros.
        """
        keys, values = zip(*param_dict.items())
        result = [dict(zip(keys, v)) for v in product(*values)]
        print(f'Hiper parâmetros montados: {result}')
        return result

    @staticmethod
    def greedy_search(model, param_grid:list, X_train:pd.Series, y_train:pd.Series, vectorizer):
        """
        Executa busca usando hiper parâmetros.
        
        Args:
            model (class): Um modelo do Sklearn [MultinomialNB | LogisticRegression | LinearSVC].
            param_grid (list): Hiper parâmetros.
            X_train (pd.Series): Dados.
            y_train (pd.Series): Classes dos dados (labels).
            vectorizer (Transformer): Vetorizador do Sklearn [CountVectorizer | TfidfVectorizer].
        
        Returns:
            pd.DataFrame: Pandas dataframe com hiper parâmetros e métricas.
        """
        results = []
        for params in param_grid:
            vec = vectorizer.fit_transform(X_train)
            clf = model(**params)
            clf.fit(vec, y_train)
            score = clf.score(vec, y_train)
            results.append({**params, 'score': score})
            print(f'Hiper parâmetro calculado para {params} com score {score}!')
        return pd.DataFrame(results)

    @staticmethod
    def metrics(vectorizer, model, test:pd.DataFrame, text_col:str='text', class_col:str='class'):
        """
        Calcula métricas no conjunto de teste usando o modelo treinado.
        
        Args:
            vectorizer (Transformer): Vetorizador usado no treinamento.
            modelo (Model): Modelo treinado.
            test (pd.DataFrame): Dados de teste.
            text_col (str): Coluna que contém texto para classificar. Valor padrão = 'text'.
            class_col (str): Coluna que contem as classes (labels). Valor padrão = 'class'.
      
        Returns:
            dict: Dicionário com as métricas f1_macro, f1_micro e acurácia.
        """
        X_test = vectorizer.transform(test[text_col])
        y_pred = model.predict(X_test)
        f1_macro = f1_score(test[class_col], y_pred, average='macro')
        f1_micro = f1_score(test[class_col], y_pred, average='micro')
        acc = accuracy_score(test[class_col], y_pred)
        result = {"f1_score_macro": f1_macro, 
                  "f1_score_micro": f1_micro,
                  "accuracy": acc}
        print(f'Métricas: {result}')
        return result
