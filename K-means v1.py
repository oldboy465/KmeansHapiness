import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import string
import numpy as np

# Cores mais distintas para os clusters
CORES_CLUSTER = ['green', 'red', 'blue', 'yellow', 'brown', 'pink', 'black', 'purple', 'orange', 'cyan']

def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    return caminho_arquivo

def carregar_dados(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    return df

def aplicar_elbow_method(df):
    # Normaliza os dados (excluindo a primeira coluna - nomes dos países)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 1:])
    
    # Calcula a inércia para diferentes números de clusters
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plotagem do gráfico de Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o', linestyle='--')
    plt.title('Método Elbow')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

def aplicar_kmeans(df, num_clusters):
    # Normaliza os dados (excluindo a primeira coluna - nomes dos países)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 1:])
    
    # Aplica o algoritmo K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Adiciona os clusters ao DataFrame original
    df['Cluster'] = clusters
    
    # Reduz a dimensionalidade para 2 componentes principais para visualização
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    # Plotagem dos clusters
    plt.figure(figsize=(10, 6))
    for i in range(num_clusters):
        plt.scatter(df_pca[clusters == i, 0], df_pca[clusters == i, 1], color=CORES_CLUSTER[i], label=f'Cluster {i+1}', edgecolor='k', s=50)
    plt.title(f'Clusters encontrados ({num_clusters} clusters)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def atribuir_nomes_clusters(df, num_clusters):
    # Cria uma lista de letras para nomear os clusters (A, B, C, ...)
    letras_clusters = list(string.ascii_uppercase)[:num_clusters]
    
    # Adiciona uma coluna ao DataFrame com os nomes dos clusters
    df['Nome Cluster'] = [letras_clusters[i] for i in df['Cluster']]
    
    return df

def exportar_dataframe(df):
    root = tk.Tk()
    root.withdraw()
    caminho_salvar = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Arquivo Excel", "*.xlsx")], title="Salvar DataFrame")
    
    if caminho_salvar:
        df.to_excel(caminho_salvar, index=False)
        print(f"DataFrame exportado com sucesso para {caminho_salvar}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    caminho_arquivo = selecionar_arquivo()
    
    if caminho_arquivo:
        df = carregar_dados(caminho_arquivo)
        
        while True:
            # Aplica o método Elbow para encontrar o número ideal de clusters
            aplicar_elbow_method(df)
            
            # Pergunta ao usuário quantos clusters deseja utilizar
            num_clusters = int(input("\nInsira o número de clusters desejado: "))
            
            # Aplica o algoritmo K-means com o número de clusters escolhido e plota os resultados
            aplicar_kmeans(df, num_clusters)
            
            # Pergunta ao usuário se deseja testar com outro número de clusters
            resposta = input("\nDeseja testar com outro número de clusters? (S/N): ").upper()
            
            if resposta != 'S':
                break
            
        # Se não for continuar testando, atribui nomes aos clusters e adiciona uma coluna no DataFrame
        df = atribuir_nomes_clusters(df, num_clusters)
        print("\nDataFrame com nomes de clusters atribuídos:")
        print(df)
        
        # Exporta o DataFrame para um arquivo Excel
        exportar_dataframe(df)
