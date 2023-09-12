import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Ler o arquivo Excel
data = pd.read_excel("barrettII_eyes_clustering.xlsx")
# Remova colunas não numéricas, como 'ID' ou outras colunas de texto que não podem ser usadas
#só tirou o id 
data = data.select_dtypes(include=[np.number])
# Verifique se existem dados faltosos (NaN)
#nao existe :)
print(data.isna().sum())


def kmeans_algo():
    #eu deixei todas as colunas, se quiserem testar, fiquem a vontade
    X = data[['AL', 'ACD', 'WTW', 'K1', 'K2']]

    # Defina o número de clusters (k)
    #o unico parametro que to mudando atualmente é o n de grupos, podem procurar outros parametros para mudar
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html esse link tem os outros atributos modifcaveis
    k = 2
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    labels = kmeans.labels_
    data['Cluster'] = labels

    # Crie DataFrames separados para cada grupo
    cluster_dataframes = []
    for cluster_label in range(k):
        cluster_df = data[data['Cluster'] == cluster_label]
        cluster_dataframes.append(cluster_df)

    # Salve cada DataFrame em um arquivo
    for i, cluster_df in enumerate(cluster_dataframes):
        #pode mudar o nome
        cluster_df.to_csv(f'Gerado pelo kmeans{i + 1}.csv', index=False)

def dbscan_algo():

    X = data[['AL', 'ACD', 'WTW', 'K1', 'K2']]

    #O Dbscan só importa os seguintes valores: eps é a distancia que ele escaneia procurando vizinhos e min_samples é o numero MINIMO de vizinhos
    #que devem ter distancia eps para serem agrupados
    dbscan = DBSCAN(eps=1.0, min_samples=10)
    dbscan.fit(X)

    # Obtenha os rótulos dos clusters para cada observação
    labels = dbscan.labels_

    # Adicione os rótulos dos clusters ao DataFrame original
    data['Cluster'] = labels

    # Divida os dados em grupos com base nos rótulos dos clusters
    groups = data.groupby('Cluster')

    # Salve cada grupo em um arquivo CSV separado
    for group_id, group_data in groups:
        group_data.to_csv(f'Gerado pelo Dbscan grupo_{group_id}.csv', index=False)


kmeans_algo()
dbscan_algo()
