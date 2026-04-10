import numpy as np 
import pandas as pd 
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus


def connect_to_database():
    name = input('Enter the database name: ')
    user = input('Enter psql user: ')
    password = input('Enter user password: ')
    port = input ('Enter database port: ')
    
    try:
        password_encoded = quote_plus(password)
        engine = create_engine(f'postgresql+psycopg2://{user}:{password_encoded}@localhost: {port}/{name}')
        print('Successful connection')
        return engine
    
    except Exception as e: 
        print('Connetion failed')
        return None

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import DBSCAN


def cluster_and_label(data, create_plot = True): 
    data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps= 0.5,min_samples= 100).fit(data)

    core_mask = np.zeros_like(db.labels_,dtype= bool)
    core_mask[db.core_sample_indices_] = True

    labels = db.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    run_metadata = {
        'n_clusters' : num_clusters,
        'n_noise' : num_noise,
        'silhouetteCoeff' : metrics.silhouette_score(data,labels= labels),
        'labels' : labels
    }


    if create_plot:
        clustering_results(data,labels,core_mask,num_clusters)


    else: 
        pass

    return run_metadata


import matplotlib.pyplot as plt 

def clustering_results (data,labels,core_mask,num_clusters):
    fig = plt.figure(figsize= (10,10))

    unique_labels = set(labels)
    colors = [plt.cm.autumn(each) for each in np.linspace(0,1,len(unique_labels))]

    for k, color in zip(unique_labels,colors):
        if k == -1: 
            color = [0,0,0,1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_mask]
        plt.plot(xy[:,0],xy[:,1],'o',mfc = tuple(color), mec = 'k', ms = 14)
        xy = data[class_member_mask & ~core_mask]
        plt.plot(xy[:,0],xy[:,1],'^',mfc = tuple(color),mec = 'k', ms = 14)
    plt.xlabel('Standard Scaled Duration')
    plt.ylabel('Standard Scaled Distance')
    plt.title(f'Estimated cluster number: {num_clusters}')
    plt.savefig('distances_and_durations.png')


import logging 
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)    

if __name__ == '__main__':
    import os
    file_path = 'uber-rides.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

    else: 
        logging.info('Getting rides information...')
        connection = connect_to_database()

        if connection is not None: 
            logging.info('Fetching data')
            df = pd.read_sql('SELECT t.duration_mins,t.distance_km,l.city FROM trips t JOIN locations l ON t.pickup_location_id = l.location_id;', connection)
            X = df[['duration_mins','distance_km','city']] 


            logging.info('Clustering and labelling...')

            results = cluster_and_label(X,create_plot= True)
            df['label'] = results ['labels']

            logging.info('Outputing to JSON...')
            df.to_json('uber-labels.json',orient= 'records')
