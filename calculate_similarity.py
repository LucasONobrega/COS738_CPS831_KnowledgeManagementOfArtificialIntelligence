import pandas as pd
import numpy as np
import ast

from sklearn.metrics.pairwise import cosine_similarity

# Função para combinar embeddings (média)
def combine_embeddings(embeddings):
    return np.mean(embeddings, axis=0)

# Agrupar dataframe pelo nome e tipo, combinando os embeddings dos documentos e palavras-chave
def combine_embeddings_by_group(df):
    grouped_df = df.groupby(['NAME', 'TYPE']).agg({
        'DOCUMENT_EMBEDDING': lambda x: combine_embeddings(x.tolist()),
        'KEYWORDS_EMBEDDING': lambda x: combine_embeddings(x.tolist())
    }).reset_index()

    return grouped_df

# Calcula similaridade
def calculate_similarity(df):
    similarity_df = pd.DataFrame(columns=['NAME','DOCUMENT_EMBEDDING_REAL','DOCUMENT_EMBEDDING_SIMULATED','DOCUMENT_SIMILARITY','KEYWORDS_EMBEDDING_REAL','KEYWORDS_EMBEDDING_SIMULATED','KEYWORDS_SIMILARITY'])
    
    for name in df['NAME'].unique():
        new_row = {}
        new_row['NAME'] = name

        new_row['DOCUMENT_EMBEDDING_REAL'] = df.loc[(df['NAME'] == name) & (df['TYPE'] == 'Real'), 'DOCUMENT_EMBEDDING'].values[0].tolist()
        new_row['DOCUMENT_EMBEDDING_SIMULATED'] = df.loc[(df['NAME'] == name) & (df['TYPE'] == 'Simulated'), 'DOCUMENT_EMBEDDING'].values[0].tolist()

        new_row['DOCUMENT_SIMILARITY'] = cosine_similarity([new_row['DOCUMENT_EMBEDDING_REAL']],[new_row['DOCUMENT_EMBEDDING_SIMULATED']])[0][0]

        new_row['KEYWORDS_EMBEDDING_REAL'] = df.loc[(df['NAME'] == name) & (df['TYPE'] == 'Real'), 'KEYWORDS_EMBEDDING'].values[0].tolist()
        new_row['KEYWORDS_EMBEDDING_SIMULATED'] = df.loc[(df['NAME'] == name) & (df['TYPE'] == 'Simulated'), 'KEYWORDS_EMBEDDING'].values[0].tolist()

        new_row['KEYWORDS_SIMILARITY'] = cosine_similarity([new_row['KEYWORDS_EMBEDDING_REAL']],[new_row['KEYWORDS_EMBEDDING_SIMULATED']])[0][0]

        new_row = pd.DataFrame([new_row])
        similarity_df = pd.concat([similarity_df, new_row], ignore_index=True)
        
    return similarity_df

if __name__ == '__main__':
    df = pd.read_csv('data/opinions_embedding.csv',sep=';')

    df['DOCUMENT_EMBEDDING'] = df['DOCUMENT_EMBEDDING'].apply(lambda x: ast.literal_eval(x))
    df['KEYWORDS_EMBEDDING'] = df['KEYWORDS_EMBEDDING'].apply(lambda x: ast.literal_eval(x))

    df = combine_embeddings_by_group(df)
    
    similarity_df = calculate_similarity(df)
    similarity_df.to_csv('data/opinions_similarity.csv', index=False, sep=';')

    print(similarity_df[['NAME','DOCUMENT_SIMILARITY']],'\n')
    print(similarity_df[['NAME','KEYWORDS_SIMILARITY']])