from builtins import breakpoint
from pydoc import doc
from unittest import result
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from bertopic import BERTopic
from umap import UMAP
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import itertools
import numpy as np
import json
from hdbscan import HDBSCAN
from keybert import KeyBERT
import pandas as pd

np.random.seed(0)
import torch
torch.manual_seed(0)
import random
random.seed(0)


import nltk
nltk.download('wordnet')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, mutual_info_score, v_measure_score, completeness_score


class TopicClustering:
    def __init__(self, config):
        self.database = config['database']
        self.stopwords = config['stopwords']
        self.demonyms = config['demonyms']
    

    def get_cluster_title(self, topics, embeddings, docs_title, topic_keywords, docs):   

        # return cluster_title
        num_topic = len(np.unique(topics)) - 1 #minus non-topic
        # num_topic = len(np.unique(topics)) # For Kmeans
        
        scores = [0] * num_topic
        cluster_title = ['none'] * num_topic


        for i, doc in enumerate(docs):
            doc_label = topics[i] # one number in range topics 
            if doc_label < 0: #Skip non-topic documents              
                continue
            score = 0
            for keyword in topic_keywords[doc_label].split('_')[1:]: # Split topic_keywords into works 
                score += doc.count(keyword) # count the number of time when word appear in document 

            # if len(doc) == 0:
            # breakpoint()

            score = score*10 /len(doc) # Calculate the average number of times the keyword appears in the entire text
            if score > scores[doc_label]: # If score greater than 0 
                scores[doc_label] = score
                cluster_title[doc_label] = topic_keywords[doc_label] + ': ' + docs_title[i]


        docs_cluster = {k: [] for k in range(-1, num_topic)}   # For HDBSCAN, OPTICS cluster model 
        docs_cluster_id = {k: [] for k in range(-1, num_topic)}

        # docs_cluster = {k: [] for k in range(0, num_topic)}    # For Kmean cluster model 
        # docs_cluster_id = {k: [] for k in range(0, num_topic)}

        for i, doc in tqdm(enumerate(docs)):

            docs_cluster[topics[i]].append(str(i) + '_' + docs_title[i])
            docs_cluster_id[topics[i]].append(i)
            
            
        result = {}
        for id, cluster in docs_cluster.items():
            cluster_info = dict(keyword=cluster_title[id].split(':')[0], cluster_title = cluster_title[id].split(':')[1], docs_title = cluster)
            result[id] = cluster_info

        return result

    def get_docs_db(self, time):
        # time format "2023-01-09"
        docs = []                # NYTnews + BBCnews
        docs_title = []          # NYTnews + BBCnews
        
        # Load NewYorkTimes News data
        path_to_NYTnews = self.database + "NYT_news/" + str(time).replace('-', '/') + "/articles"
        with open(path_to_NYTnews, 'r') as f:
            data = json.loads(f.read())
            
        for item in tqdm(data["articles"]):
            if len(item['content']) != 0:
                docs.append(item['content'])
                docs_title.append(item['title'])
        print("The number of NYT news : ", str(time), len(docs))   
            
        # Load NewYorkTimes News data
        path_to_BBCnews = self.database + "BBC_news/" + str(time).replace('-', '/') + "/articles"
        with open(path_to_BBCnews, 'r') as f:
            data = json.loads(f.read())
            
        for item in tqdm(data["articles"]):
            if len(item['content']) != 0:
                docs.append(item['content'])
                docs_title.append(item['title'])
                
        print("The number of BBC news : ", str(time), len(docs))  
        return docs, docs_title


    def preprocessing(self, docs):

        # Split the documents into tokens.
        print("TOKENIZING ....")
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        print("REMOVING SHORT TOKEN ....")
        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 1] for doc in docs]

        # Lemmatize the documents, normalize to raw format 
        print("LEMMATIZING ....")
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

        print("DEMONYMS ....")
        demonyms = pd.read_csv(self.demonyms)
        demonyms = dict(zip(demonyms.iloc[:,0].to_list(), demonyms.iloc[:,1].to_list()))
        docs = [[demonyms.get(token, token) for token in doc] for doc in docs]

        # Remove Stopword
        print("REMOVING STOPWORD ....")
        stopwords = set(map(str.strip, open(self.stopwords).readlines()))
        stopwords = set([i.lower() for i in stopwords])
        
        docs = [[token for token in doc if token.lower() not in stopwords] for doc in docs]
        docs = [" ".join(doc) for doc in docs]
        return docs

    def doc_embeding(self, docs):
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings_bert = sentence_model.encode(docs, show_progress_bar=True)
        return embeddings_bert

    def title_clustering(self, docs, docs_title, is_evaluation = False):
        embeddings = self.doc_embeding(docs)
        cluster_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # cluster_model = OPTICS(metric = 'cosine', min_cluster_size = 2)
        # cluster_model = KMeans(n_clusters=20)
        print("cluster_model : ", cluster_model)

        topic_model = BERTopic(hdbscan_model=cluster_model, top_n_words = 10).fit(docs, embeddings) # change top_word to top_n_words
        topics, probs = topic_model.fit_transform(docs)
        # print("topics : ", topics)

        # Change topic_model.topic_labels_ to center's title
        topic_keywords = topic_model.topic_labels_  # The default labels for each topic.
        # print("topic_keywords : ", topic_keywords)
        
            
        cluster_title = self.get_cluster_title(topics, embeddings, docs_title, topic_keywords, docs)
        
        # breakpoint()
        topic_model.set_topic_labels([k['keyword'] for i,k in cluster_title.items()])
        reduced_embeddings = UMAP(n_neighbors=20, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        fig = topic_model.visualize_documents(docs_title, reduced_embeddings=reduced_embeddings, custom_labels = True)
        fig.show()

        return cluster_title
    
    def run(self, times):
        docs = []
        docs_title = []
        for time in times:
            print(time)
            docs_per_day, docs_title_per_day = model.get_docs_db(time)
            print("the number of news in ", str(time) , len(docs_per_day))
            print("the number of news title in ", str(time) , len(docs_title_per_day))
            docs.extend(docs_per_day)
            docs_title.extend(docs_title_per_day)
        
        print("Total number of news : " , len(docs))
        print("Total number of news title : " , len(docs_title))

        docs = model.preprocessing(docs)
        result = model.title_clustering(docs, docs_title)
        
        with open("log_data/docs_cluster_{}_{}.json".format(times[0], times[-1]), "w") as outfile: 
            json.dump(result, outfile)
    
            
if __name__ == "__main__":
    cfg = {
        'database': '/home/tuanld/AIC/Nga/HotEventDetection/news-crawler-master/Result_news/',
        'stopwords': "stopword.txt",
        'demonyms' : "demonyms.csv"
    }

    # time format yyyy/mm/dd
    times = ["2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14", "2023-01-15"]
    model = TopicClustering(cfg)
    model.run(times)


'''
Run : pass tuanld_v0.5
conda activate taidvt 
cd /home/tuanld/AIC/Nga/HotEventDetection/hoteventdetection-develop/model
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$"/home/tuanld/miniconda3/lib/"
CUDA_VISIBLE_DEVICES=3 python /home/tuanld/AIC/Nga/HotEventDetection/hoteventdetection-develop/model/main.py
'''

