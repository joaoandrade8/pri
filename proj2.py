import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

#A

def sentence_clustering(d, I=None, args=None):
    score=0
    label =[]
    scores = []
    with open(d, "r") as doc:
        doc.readline()
        txt = doc.read()
        collection = nltk.sent_tokenize(txt)
        vectorizer = TfidfVectorizer(use_idf=False)
        vectorspace = vectorizer.fit_transform(collection)
        min_cluster = 2
        max_cluster = len(collection)//2
        cluster_range = range(min_cluster, max_cluster +1)
    for cluster in cluster_range:
        clustering = AgglomerativeClustering(n_clusters=cluster, linkage="average", metric='cosine').fit(vectorspace.toarray())
        silh = silhouette_score(vectorspace, clustering.labels_, metric="cosine")
        scores.append(silh)
        if silh > score:
            score = silh
            label = clustering.labels_
    return label, cluster_range, scores, min_cluster, max_cluster
    

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
    
    term_scores = np.sum(vectorspace, axis=0)
    term_scores = np.asarray(term_scores).ravel()
    top_term_indices = np.argsort(term_scores)[::-1]
    
    vocabulary = np.array(vectorizer.get_feature_names_out())
    top_terms = vocabulary[top_term_indices]
    
    return set(top_terms[:n_keywords])


import matplotlib.pyplot as plt

# Run sentence clustering and summarization
c, r, scores, min_cluster, max_cluster = sentence_clustering("001.txt")
s= summarization("001.txt", c)


#B
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def feature_extraction(s, d, args=None):
    """
    Extracts features of potential interest considering the characteristics of the sentence and the wrapping document.

    Args:
    s (str): The sentence.
    d (str): The enclosed document.
    args (dict): Additional arguments.

    Returns:
    list: Sentence-specific feature vector.
    """
    with open(d, "r") as doc:
        
        d = doc.read()
        
        # Initialize an empty feature vector
        feature_vector = []

        # Feature 1: Length of the sentence
        feature_vector.append(len(s))

        # Feature 2: Position of the sentence in the document
        sentence_index = d.find(s)
        feature_vector.append(sentence_index)

        # Feature 3: Number of words in the sentence
        num_words = len(s.split())
        feature_vector.append(num_words)

        # Feature 4: TF-IDF of the sentence in the document
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([d, s])
        tfidf_sentence = tfidf_matrix[1].toarray()[0]  # Extract TF-IDF vector for the sentence
        feature_vector.append(np.mean(tfidf_sentence))  # Using mean TF-IDF value as the feature

        # Feature 5: Sentence position in the paragraph
        sentence_position = d[:sentence_index].count('.') / max(d.count('.'), 1)  # Normalize by total number of sentences
        feature_vector.append(sentence_position)

        # Feature 6: Presence of numeric characters in the sentence
        has_numeric = any(char.isdigit() for char in s)
        feature_vector.append(int(has_numeric))  # Convert boolean to integer 

    return feature_vector

def create_training_data(documents_folder, references_folder):
    """
    Create training document collection (Dtrain) and reference extracts (Rtrain) from text files in folders.

    Args:
    documents_folder (str): Path to the folder containing training documents.
    references_folder (str): Path to the folder containing reference extracts.

    Returns:
    tuple: Tuple containing Dtrain (list of document contents) and Rtrain (list of reference extract contents).
    """
    Dtrain = []
    Rtrain = []

    # Read training documents
    for filename in os.listdir(documents_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(documents_folder, filename), 'r', encoding='utf-8') as file:
                document_content = file.read()
                Dtrain.append(document_content)

    # Read reference extracts
    for filename in os.listdir(references_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(references_folder, filename), 'r', encoding='utf-8') as file:
                reference_content = file.read()
                Rtrain.append(reference_content)

    return Dtrain, Rtrain

def training(Dtrain, Rtrain, args=None):
    """
    Learns a classifier to predict the presence of a sentence in the summary.

    Args:
    Dtrain (list): Training document collection.
    Rtrain (list): Reference extracts for the training documents.
    args (dict, optional): Optional arguments related to the classification process.

    Returns:
    sklearn.naive_bayes.MultinomialNB: Trained classification model.
    """
    vectorizer = TfidfVectorizer(use_idf=False)
    trainvec = vectorizer.fit_transform(Dtrain)

    classifier = MultinomialNB() 
    classifier.fit(trainvec, Dtrain)

    return classifier

def summarization(d, M, p=None, l=None, args=None):
    """
    Summarizes document d without prefixed size limits or with prefixed size limits by identifying relevant sentence candidates for the summary.

    Args:
    d (str): Document to be summarized.
    M (sklearn.naive_bayes.MultinomialNB): Trained classification model.
    p (int, optional): Maximum number of sentences.
    l (int, optional): Maximum number of characters.
    args (dict, optional): Additional arguments.

    Returns:
    str: Summary of document d.
    """
    # Load the document
    with open(d, "r") as file:
        document_content = file.read()

    # Tokenize the document into sentences
    sentences = nltk.sent_tokenize(document_content)

    # Extract features from each sentence using the same vectorizer as during training
    tfidf_matrix = vectorizer.transform(sentences)

    # Predict relevance scores for each sentence using the classification model
    relevance_scores = M.predict(tfidf_matrix)

    # Sort sentences based on relevance scores
    sorted_sentences_indices = sorted(range(len(relevance_scores)), key=lambda k: relevance_scores[k], reverse=True)
    sorted_sentences = [sentences[i] for i in sorted_sentences_indices]

    # Construct the summary based on size limits if provided
    if p is not None or l is not None:
        summary_sentences = []
        total_chars = 0
        for sentence in sorted_sentences:
            if p is not None and len(summary_sentences) >= p:
                break
            if l is not None and total_chars + len(sentence) > l:
                break
            summary_sentences.append(sentence)
            total_chars += len(sentence)
        summary = ' '.join(summary_sentences)
    else:
        # Construct the summary without size limits
        summary = ' '.join(sorted_sentences)

    return summary

"""
documents_folder = "BBC News Summary/News Articles/politics"
references_folder = "BBC News Summary/Summaries/politics"

Dtrain, Rtrain = create_training_data(documents_folder, references_folder)

# Fit TF-IDF vectorizer with training data
vectorizer = TfidfVectorizer(use_idf=False)
vectorizer.fit(Dtrain)

# Train the classification model
model = training(Dtrain, Rtrain)

# Example usage
Dtest = "413.txt"  # Path to the test document
import os


from sklearn.metrics import precision_score, recall_score, f1_score

def docs_in_folder(loc):
    text_documents = []
    for root, dirs, files in os.walk(loc):
        for file in files:
            text_documents.append(root+'/'+file)
    return text_documents

def evaluation(S, Rset, args=None):
    evaluation_stats = {}

    precision = len(S.intersection(Rset)) / len(S) 
    recall = len(S.intersection(Rset)) / len(Rset)
    f_measure = (2 * (precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
    evaluation_stats['Precision'] = precision
    evaluation_stats['Recall'] = recall

    evaluation_stats['F_measure'] = f_measure

    #average_precision = sum([len(S.intersection(Rset[:i+1])) / (i+1) for i in range(len(Rset))]) / len(Rset)
    #evaluation_stats['MAP'] = average_precision

    return evaluation_stats

def supervised_evaluation(Dtest_folder, Rtest_folder, M, args=None):
    evaluation_scores = {}
    
    # Read the contents of testing documents and reference extracts
    Dtest = docs_in_folder(Dtest_folder)
    Rtest = docs_in_folder(Rtest_folder)
    
    for e, d in enumerate(Dtest):
        s = summarization(d, M, p=6)
        with open (Rtest[e], "r") as doc:
            r = doc.read()
                    
        s_sentences = set(nltk.word_tokenize(s))
        r_sentences = set(nltk.word_tokenize(r))


        scores = evaluation(s_sentences, r_sentences)
        evaluation_scores[d] = scores
    
    return evaluation_scores

evaluation_scores = supervised_evaluation(documents_folder, references_folder, model)
print(evaluation_scores)


"""

