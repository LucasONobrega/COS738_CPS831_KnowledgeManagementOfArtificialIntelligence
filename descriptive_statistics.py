import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

if __name__ == '__main__':
    df = pd.read_csv('data/opinions_similarity.csv',sep=';')

    df['DOCUMENT_EMBEDDING_REAL'] = df['DOCUMENT_EMBEDDING_REAL'].apply(lambda x: ast.literal_eval(x))
    df['DOCUMENT_EMBEDDING_SIMULATED'] = df['DOCUMENT_EMBEDDING_SIMULATED'].apply(lambda x: ast.literal_eval(x))
    df['KEYWORDS_EMBEDDING_REAL'] = df['KEYWORDS_EMBEDDING_REAL'].apply(lambda x: ast.literal_eval(x))
    df['KEYWORDS_EMBEDDING_SIMULATED'] = df['KEYWORDS_EMBEDDING_SIMULATED'].apply(lambda x: ast.literal_eval(x))

    print('DOCUMENT SIMILARITY:')
    print(df[['NAME','DOCUMENT_SIMILARITY']])
    print('\nDescriptive statistics:') # Estatísticas descritivas
    print(df['DOCUMENT_SIMILARITY'].describe())
    
    print('\n\nKEYWORDS SIMILARITY:')
    print(df[['NAME','KEYWORDS_SIMILARITY']])
    print('\nDescriptive statistics:') # Estatísticas descritivas
    print(df['KEYWORDS_SIMILARITY'].describe())

    # Box Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([df['DOCUMENT_SIMILARITY'], df['KEYWORDS_SIMILARITY']], labels=['DOCUMENT SIMILARITY', 'KEYWORDS SIMILARITY'])
    plt.title('Box Plot of Document and Keyword Similarities')
    plt.ylabel('Similarity')
    plt.show()