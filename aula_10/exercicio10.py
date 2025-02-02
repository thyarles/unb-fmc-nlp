# Você vai precisar desses pacotes do Python (versões no requirements.txt)
# pandas scikit-learn transformers torch matplotlib

# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from torch.utils.data import Dataset

class MyBert:
    def __init__(self, modelo):
        # Modelo
        self.modelo = modelo
        self.modelo_treinado = None
        # Tokenizador
        self.tokenizer = BertTokenizer.from_pretrained(self.modelo)
        self.max_length = 128
        # Dados
        self.data = None
        self.treino = None
        self.teste = None
        self.validacao = None
        self.le = LabelEncoder()
        # Dataset
        self.ds_treino = None
        self.ds_teste = None
        self.ds_validacao = None


    # Cria dataset
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
    # Carrega e divide os dados    
    def carregar_csv(self, csv_file):
        # Carrega dados
        self.data = pd.read_csv(csv_file).drop(columns=['file_name'])
        self.data['label'] = self.le.fit_transform(self.data['class'])

        # Faz a divisão em 70% treino, 10% validação e 20% teste
        self.treino, df = train_test_split(self.data, test_size=0.3, stratify=self.data['label'])
        self.teste, self.validacao = train_test_split(df, test_size=0.333, stratify=df['label'])

        # Cria datasets
        self.ds_treino = self.TextDataset(
            self.treino['text'].tolist(),
            self.treino['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        self.ds_validacao = self.TextDataset(
            self.validacao['text'].tolist(),
            self.validacao['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        self.ds_teste = self.TextDataset(
            self.teste['text'].tolist(),
            self.teste['label'].tolist(),
            self.tokenizer,
            self.max_length
        )

    # Exibe amostra (requisito do exercício)
    def imprimir_amostra(self):
        amostra = self.treino['text'].iloc[0]
        print("\nTexto de amostra  :", amostra)
        print("Amostra em tokens:", self.tokenizer.tokenize(amostra))
        print("Tokens IDs       :", self.tokenizer.encode(amostra, add_special_tokens=True))


    # Exibe matriz de confusão
    def matriz_confusao(self, resultado):    
        preds = np.argmax(resultado.predictions, axis=1)
        cm = confusion_matrix(resultado.label_ids, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.le.classes_)
        disp.plot(xticks_rotation='vertical')
        plt.title('Matriz de confusão')
        plt.show()


    # Cálculo das métricas
    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_macro': f1_score(labels, preds, average='macro')
        }


    # Iniciar o treino
    def treinar(self, argumentos_treino):
        # Inicializa o modelo
        modelo = BertForSequenceClassification.from_pretrained(
            self.modelo,
            num_labels=len(self.le.classes_)
        )

        # Cria o treino
        self.modelo_treinado = Trainer(
            model=modelo,
            args=argumentos_treino,
            train_dataset=self.ds_treino,
            eval_dataset=self.ds_validacao,
            compute_metrics=self.compute_metrics,
        )

        # Treina o modelo
        self.modelo_treinado.train()        