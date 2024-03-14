import os, os.path
from whoosh import index, fields, scoring
from whoosh.fields import Schema, TEXT
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser
from whoosh.query import Term, And
import time
import nltk
from collections import defaultdict

#global vars
index_location = "index"

#create schema
def create_schema():
    if not os.path.exists(index_location): os.mkdir(index_location)
    schema = fields.Schema(id = fields.NUMERIC(stored=True), content=fields.TEXT)
    index.create_in(index_location, schema)

def docs_in_folder(loc):
    text_documents = []
    for root, dirs, files in os.walk(loc):
        for file in files:
            text_documents.append(root+'/'+file)
    return text_documents

#add docs 
def add_to_schema_by_phrase(text):
    ix = open_dir(index_location)
    writer = ix.writer()
    with open(text, "r") as doc:
            doc.readline()
            text = doc.read()
            phrases = text.split('.')
            for el in phrases:
                writer.add_document(id=1, content=el)
    writer.commit()
    return phrases

def add_to_schema_by_word(text):
    ix = open_dir(index_location)
    writer = ix.writer()
    with open(text, "r") as doc:
        doc.readline()
        text = doc.read()
        words = nltk.word_tokenize(text)
        for el in words:
            writer.add_document(id=1, content=el)
    writer.commit()
    return words 

#query the schema
def query_schema(query, method):
    if method == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
        bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        return list(get_bert_output(tokenizer, bert_model, query))
    if method == "tfidf":
        score = scoring.TF_IDF()
    elif method == "bm25":
        score = scoring.BM25F() #default values are the required
    ix = index.open_dir(index_location)
    with ix.searcher(weighting=score) as searcher:
        q = QueryParser('content', ix.schema, group=OrGroup).parse(query)
        return searcher.search(q, limit=100)

"""indexing function as requested"""
def indexing(D, args=None):
    start = time.time()
    create_schema()
    docs = docs_in_folder(D)
    i = 1
    for el in docs:
        add_to_schema_by_phrase(el)
    end = time.time()    
    return index_location, round(end - start,4)

"""summarization function as requested"""
def summarization(d, p, l,o, I, model, args=None):
    try:
        ix = index.open_dir(I)
    except index.EmptyIndexError: 
        I, _ = indexing(I)  # Reindex the collection if the index is empty
    ix = index.open_dir(I) 
    relevance_scores = {}
    phrases = add_to_schema_by_phrase(d)
    for i, sentence in enumerate(phrases):
        results = query_schema(sentence, model)
        for doc, score in results.items():
            relevance_scores[i] = score if score else 0 
        
    sorted_sentences = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=(o == 'relevance')) #if relevance
    
    selected_sentences = sorted_sentences[:p] if l is None else [pair for pair in sorted_sentences if len(pair[0]) <= l]
    return selected_sentences
    
"keyword_extraction"
def keyword_extraction(d,p,I, model, args = None):
    try:
        ix = index.open_dir(I)
    except index.EmptyIndexError: 
        I, _ = indexing(I)  # Reindex the collection if the index is empty
    ix = index.open_dir(I) 
    relevance_scores = defaultdict(float)
    words = add_to_schema_by_word(d)
    for word in words:
        results = query_schema(word, model)
        for doc, score in results.items():
            relevance_scores[word] += score if score else 0
    sorted_keywords = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    selected_keywords = tuple(keyword for keyword, score in sorted_keywords[:p])
    return selected_keywords

"evaluation"
def evaluation(S, Rset, args=None):
    evaluation_stats = {}
    with open(S, "r") as doc:
        txt = doc.read()
        Sset = set(nltk.word_tokenize(txt))
    precision = len(Sset.intersection(Rset)) / len(Sset) 
    recall = len(Sset.intersection(Rset)) / len(Rset)
    f_measure = (2 * (precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
    evaluation_stats['Precision'] = precision
    evaluation_stats['Recall'] = recall

    evaluation_stats['F_measure'] = f_measure

    average_precision = sum([len(Sset.intersection(Rset[:i+1])) / (i+1) for i in range(len(Rset))]) / len(Rset)
    evaluation_stats['MAP'] = average_precision

    return evaluation_stats

#bert
import torch
from transformers import BertModel, BertTokenizer

def get_bert_output(tokenizer, model, sentence, mode='cls', optype='sumsum'):
    tokenized_text = tokenizer.tokenize(sentence)
    tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    segments_tensors = torch.tensor([[1] * len(tokenized_text)])
    outputs = model(tokens_tensor, segments_tensors)
    if mode == 'cls': embedding = outputs["last_hidden_state"].squeeze()[0]
    elif mode == 'pooled': embedding = outputs["pooler_output"].squeeze()
    else: #'hidden'
        layers = torch.stack(outputs['hidden_states'][-4:])
        if optype == "sumsum": embedding = torch.sum(layers.sum(0).squeeze(), dim=0) 
        elif optype == "summean": embedding = torch.sum(layers.mean(0).squeeze(), dim=0) 
        elif optype == "meanmean": embedding = torch.mean(layers.mean(0).squeeze(), dim=0)    
        else: embedding = torch.mean(layers.sum(0).squeeze(), dim=0)  
    return embedding.detach().numpy()[0]

if __name__ == "__main__":
    s = summarization("t1.txt", 10, None, None, "text", "tfidf")
    print(evaluation("s1.txt", s, None))
    