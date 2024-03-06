import os, os.path
from whoosh import index, fields, scoring
from whoosh.fields import Schema, TEXT
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser
from whoosh.query import Term, And
import time
import nltk

#global vars
index_location = "/index"


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
def add_doc_to_schema(text, num):
    ix = open_dir(index_location)
    writer = ix.writer()
    writer.add_document(id=num, content=text)
    writer.commit()

#query the schema
def query_schema(query):
    ix = index.open_dir(index_location)
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        q = QueryParser('content', ix.schema, group=OrGroup).parse(query)
        results = searcher.search(q, limit=100)
        for r in results: print(r)

#preprocess text
def preprocess(text:str):
    tokens = nltk.word_tokenize(text)
    new_tokens = [nltk.lemmatizer.lemmatize(token) for token in tokens]
    return new_tokens

"""indexing function as requested"""
def indexing(D, args=None):
    start = time.time()
    create_schema()
    docs = docs_in_folder(D)
    i = 0
    for el in docs:
        with open(el, "r") as doc:
            text = preprocess(doc.read())
            add_doc_to_schema(text, i)
        i+=1
    end = time.time()    
    return index_location, end - start

"""summarization function as requested"""
def summarization(d, p, l,o, I, args=None):
    if not index.open_dir(I): #I is not an index, its a D (collection)
        I,_ = indexing(I)
    ix = index.open_dir(I)
    searcher = ix.searcher(weighting=scoring.TF_IDF())
    relevance_scores = {}
    with open(d, "r") as doc:
        text = preprocess(doc.read())
        phrases = text.split('.')
        for i, sentence in enumerate(phrases):
            query = QueryParser('content', ix.schema).parse(sentence)
            results = searcher.search(query, limit=1)
            relevance_scores[i] = results.score if results else 0
        
    sorted_sentences = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=(o == 'relevance')) #if relevance
    
    selected_sentences = sorted_sentences[:p] if l is None else [pair for pair in sorted_sentences if len(pair[0]) <= l]
    return selected_sentences

    
"keyword_extraction"
def keyword_extraction(d,p,I,args):
    return

#bert
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

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
    return embedding.detach().numpy()
    