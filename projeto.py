import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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
summarized_text = summarization(Dtest, model, p=3)
print(summarized_text)
