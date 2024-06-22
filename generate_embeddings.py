import pandas as pd
import numpy as np
import ast

from openai import OpenAI
from keybert import KeyBERT

import nltk
# nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import spacy
nlp = spacy.load("pt_core_news_lg")

# Gerador de embedding usando API da OpenAI
def get_embedding(text):
    client = OpenAI(
    api_key='OPENAI_API_KEY',
    )

    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    
    return response.data[0].embedding

# Dividir texto em sentenças
def split_sentences(text):
    return sent_tokenize(text, language='portuguese')

# Função para combinar embeddings (média)
def combine_embeddings(embeddings):
    return np.mean(embeddings, axis=0)

# Calcular embeddings dos documentos
def document_embedding(df):
    documents_embeddings = []
    for name,text in zip(df['NAME'],df['TEXT']):
        sentences = split_sentences(text)
        
        i = 1
        sentences_embeddings = []
        for s in sentences:
            print(f'{name} {i}/{len(sentences)}')
            i+=1
            se = get_embedding(s)
            sentences_embeddings.append(se)

        de = combine_embeddings(sentences_embeddings)
        de = de.tolist()
        documents_embeddings.append(de)

    return documents_embeddings

# Extrair palavras-chave
def extract_keywords(df):
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = []
    for text in df['TEXT']:
      keyword = model.extract_keywords(text, top_n = 20)
      keyword = filter_keyword(keyword)
      keywords.append(keyword)

    return keywords

# Filtrar palavras-chave
def filter_keyword(keywords):
    stop_words = set(stopwords.words('portuguese'))
    filtered = []
    
    for keyword,score in keywords:
        doc = nlp(keyword)
        token = doc[0]
        
        if len(token.text) < 5 or token.text.lower() in stop_words or token.lemma_ in filtered:
            continue
        
        filtered.append(token.lemma_)
        
        if len(filtered) == 10:
            break
    
    return filtered

# Calcular embeddings das palavras-chave
def keyword_embedding(df):
    keywords_embeddings = []
    for keyword in df['KEYWORDS']:
        words_embeddings = []
        for word in keyword:
            we = get_embedding(word)
            words_embeddings.append(we)

        kwe = combine_embeddings(words_embeddings)
        kwe = kwe.tolist()
        keywords_embeddings.append(kwe)

    return keywords_embeddings

if __name__ == '__main__':
    df = pd.read_csv('data/opinions_clean.csv',sep=';')

    df['DOCUMENT_EMBEDDING'] = document_embedding(df)
    df['KEYWORDS'] = extract_keywords(df)
    df['KEYWORDS_EMBEDDING'] = keyword_embedding(df)

    df.to_csv('data/opinions_embedding.csv', index=False, sep=';')