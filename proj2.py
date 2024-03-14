from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import nltk

#A

def sentence_clustering(d, I=None, args=None):
    score=0
    label =[]
    with open(d, "r") as doc:
        doc.readline()
        txt = doc.read()
        collection = nltk.sent_tokenize(txt)
        vectorizer = TfidfVectorizer(use_idf=False)
        vectorspace = vectorizer.fit_transform(collection)
        min_cluster = 2
        max_cluster = len(collection)//2
        cluster_range =range(min_cluster, max_cluster +1)
    for cluster in cluster_range:
    #para escolher nr de clsuters testar com varios vs silhouette, valor max é o ideal
        clustering = AgglomerativeClustering(n_clusters=cluster, linkage="average", metric='cosine').fit(vectorspace.toarray())
        silh = silhouette_score(vectorspace, clustering.labels_, metric="cosine")
        if silh > score:
            score = silh
            label = clustering.labels_
    return label

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, scores, marker='o', linestyle='-')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(np.arange(min_cluster, max_cluster + 1, step=1))
    plt.grid(True)
    plt.show()
    
c = sentence_clustering("413.txt")

def summarization(d, C, I=None, args=None): #usar f1
    n_clusters = max(C)+1
    centroids = []
    aux = []
    with open(d, "r") as doc:
        doc.readline()
        txt = doc.read()
        collection = nltk.sent_tokenize(txt)
        vectorizer = TfidfVectorizer(use_idf=False)
        vectorspace = vectorizer.fit_transform(collection)
    for c in range(n_clusters):
        for l in range(len(C)):
            if c == C[l]:
                aux.append(vectorspace.toarray()[l])
        cluster_vectors = np.array(aux)
        centroids.append(np.mean(cluster_vectors, axis=0))
        aux = []
    
    closest_sentences = []
    
    for cluster_id in range(n_clusters):
        distances = np.linalg.norm(cluster_vectors - centroids[cluster_id], axis=1)
        closest_index = np.argmin(distances)
        closest_sentence = collection[closest_index]
        closest_sentences.append(closest_sentence)
    
    return closest_sentences #pode ser lista??
    
def keyword_extraction(d, C, I=None, args=None):
    n_keywords=7
    with open(d, "r") as doc:
        doc.readline()
        txt = doc.read()
        collection = nltk.sent_tokenize(txt)
        vectorizer = TfidfVectorizer(use_idf=True)
        vectorspace = vectorizer.fit_transform(collection)
    
    # Sum the TF-IDF scores across all sentences for each term
    term_scores = np.sum(vectorspace, axis=0)
    term_scores = np.asarray(term_scores).ravel()
    
    # Get the indices of the terms with the highest sum
    top_term_indices = np.argsort(term_scores)[::-1]
    
    # Get the actual terms from the vectorizer vocabulary
    vocabulary = np.array(vectorizer.get_feature_names_out())
    top_terms = vocabulary[top_term_indices]
    
    return set(top_terms[:n_keywords])
    
print(keyword_extraction("413.txt", c))