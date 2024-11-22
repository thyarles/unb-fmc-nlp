import os
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score


class AtividadeTres:
    """
    Classe para buscar hiper-parâmetros, treinar o modelo, avaliar, salvar e carregar os resultados.
    """

    def __init__(self, random_state: int = None):
        """
        Inicializa a classe configurando randomizador para possibilitar reprodução.

        Args:
            random_state (int): Semente para reprodução. Valor padrão = None.
        """
        self.random_state = random_state
        print(
            "Classe iniciada sem semente!"
            if random_state == None
            else f"Classe iniciada com semente {random_state}!"
        )

    @staticmethod
    def load_data(filename: str, path: str = "in/"):
        """
        Lê dados CSV.

        Args:
            filename (str): Nome do arquivo CSV.
            path (str): Caminho para o arquivo CSV.

        Returns:
            pd.DataFrame: Arquivo convertido em Pandas.
        """
        return pd.read_csv(f"{path}/{filename}")

    def split_data(
        self, data: pd.DataFrame, text_col: str, class_col: str, test_size: float = 0.2
    ):
        """
        Divide os dados em treino e teste.

        Args:
            data (pd.DataFrame): Dados para dividir.
            text_col (str): Coluna com o texto para classificar.
            class_col (str): Coluna com a classificação.
            test_size (float): Tamanho da base de teste. Valor padrão = 0.2.

        Returns:
            pd.DataFrame: Base de treino.
            pd.DataFrame: Base de teste
            dict: Train-test splits (X_train, X_test, y_train, y_test).
        """
        X = data[text_col]
        y = data[class_col]

        if self.random_state:
            print(
                f"Dados divididos com tamanho do teste {test_size:0.2f} e semente {self.random_state}!"
            )
            xtr, xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        else:
            print(
                f"Dados divididos com tamanho do teste {test_size:0.2f} e sem semente!"
            )
            xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size)
        train = pd.DataFrame({"text": xtr, "class": ytr})
        test = pd.DataFrame({"text": xte, "class": yte})
        series = {"X_train": xtr, "X_test": xte, "y_train": ytr, "y_test": yte}
        print(f"Dicionário de séries: {series.keys()}")
        return train, test, series

    @staticmethod
    def save_data_frame(data: pd.DataFrame, filename: str, path: str = "out/"):
        """
        Salva dados treinados para o disco.

        Args:
            data (pd.DataFrame): Dados para salvar.
            file (str): Nome para o arquivo.
            path (str): Diretório para salvar. Valor padrão = 'out/'.
        """
        os.makedirs(path, exist_ok=True)
        data.to_csv(os.path.join(path, filename), index=False)
        print(f"Arquivo {path}{filename} salvo com sucesso!")

    @staticmethod
    def create_param_grid(param_dict: dict):
        """
        Monta conjunto de hiper parâmetros para uso no greedy_search.

        Args:
            param_dict (dict): Dicionário com parâmetros e valores.

        Returns:
            list: Lista de dicionário com a combinação dos hiper parâmetros.
        """
        keys, values = zip(*param_dict.items())
        result = [dict(zip(keys, v)) for v in product(*values)]
        print(f"Hiper parâmetros montados: {result}")
        return result

    @staticmethod
    def greedy_search(
        model_name,
        param_grid: list,
        series: dict,
    ):
        """
        Executa busca usando hiper parâmetros.

        Args:
            model_name (str): Um modelo do Sklearn [MultinomialNB, LogisticRegression ou LinearSVC].
            param_grid (list): Hiper parâmetros.
            series (dict): Dicionário com 'X_train', 'X_test', 'y_train', 'y_test' (pd.Series).

        Returns:
            pd.DataFrame: Pandas data frame com hiper parâmetros e métricas.
        """

        # Para eliminar warnings de convergência e configurações incompatíveis
        import warnings
        warnings.simplefilter("ignore")
        # from sklearn.exceptions import ConvergenceWarning
        # warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Apenas modelos habilitados
        if model_name == "MultinomialNB":
            model_name = MultinomialNB
        elif model_name == "LogisticRegression":
            model_name = LogisticRegression
        elif model_name == "LinearSVC":
            model_name = LinearSVC
        else:
            raise Exception("Valores válidos para modelo: MultinomialNB, LogisticRegression ou LinearSVC.")

        results = []
        vectorizer = CountVectorizer()
        X_vec_train = vectorizer.fit_transform(series['X_train'])

        for params in param_grid:
            try:
                clf = model_name(**params)
                clf.fit(X_vec_train, series['y_train'])
                score = clf.score(X_vec_train, series['y_train'])
                if score < .998: # Para evitar Overfitting
                    results.append({**params, "score": score})
            except:
                continue                
        return pd.DataFrame(results)

    @staticmethod
    def metrics(
        vectorizer,
        model,
        test: pd.DataFrame,
        text_col: str = "text",
        class_col: str = "class",
    ):
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
        f1_macro = f1_score(test[class_col], y_pred, average="macro")
        f1_micro = f1_score(test[class_col], y_pred, average="micro")
        acc = accuracy_score(test[class_col], y_pred)
        result = {
            "f1_score_macro": f1_macro,
            "f1_score_micro": f1_micro,
            "accuracy": acc,
        }
        print(f"Métricas: {result}")
        return result


# Debug
if __name__ == "__main__":
    nb_params = {   # MultinomialNB
    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0],
    'fit_prior': [True, False]
    }

    lr_params = {  # LogisticRegression
        "C": [0.1, 0.25, 1.0, 5.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    }
    svm_params = {  # SVM
        "C": [0.01, 0.1, 1, 5, 10, 50, 100],
        "loss": ["hinge", "squared_hinge"],
        "max_iter": [100, 250, 500, 1000, 5000],
    }
    a3 = AtividadeTres()
    BASE = "CSTR"
    FILENAME = BASE + ".csv"
    cstr = a3.load_data(FILENAME)
    train, test, series = a3.split_data(
        data=cstr, text_col="text", class_col="class", test_size=0.2
    )
    a3.save_data_frame(train, BASE + "_train.csv")
    a3.save_data_frame(test, BASE + "_test.csv")
    tmp = a3.greedy_search(
        model_name="nb",
        param_grid=a3.create_param_grid(nb_params),
        series=series
    )
    print(tmp)
