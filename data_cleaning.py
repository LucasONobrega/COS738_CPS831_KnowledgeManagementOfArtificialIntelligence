import pandas as pd
import re

import xmltodict
from lxml import etree

# Ler um arquivo em formato XML
def read_xml(xml_file):
    parser = etree.XMLParser()
    tree = etree.parse(f'{xml_file}', parser)
    dict = xmltodict.parse(etree.tostring(tree))
    
    return dict['FILE']['PERSONA']
    # list = [ { 'NAME'     : STRING
    #            'INFO'     : STRING
    #            'OPINIONS' : { 'Opinion' : [ { 'Type'      : STRING
    #                                           'Source'    : STRING 
    #                                           'Text'      : STRING }, ... ]
    #                         } 
    #          }, ... 
    #        ]

# Normalização
def normalize(text):
    # Remover quebras de linha e tabulações
    text = re.sub(r'[\n\t]', ' ', text)
    # Remover múltiplos espaços em branco
    text = re.sub(r' +', ' ', text)
    # Remover caracteres especiais e símbolos desnecessários
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s.,!?\'"-]', '', text)
    # Remover espaços em branco no início e no final
    text = text.strip()
    
    return text

if __name__ == '__main__':
    xml = read_xml('data/opinions.xml')

    df = pd.DataFrame(xml)
    df['OPINIONS'] = df['OPINIONS'].apply(lambda x: x['Opinion'])
    df = df.explode('OPINIONS')
    df['TYPE'] = df['OPINIONS'].apply(lambda x: x['Type'])
    df['SOURCE'] = df['OPINIONS'].apply(lambda x: x['Source'])
    df['TEXT'] = df['OPINIONS'].apply(lambda x: x['Text'])
    df.drop(columns=['OPINIONS'], inplace=True)

    df['TEXT'] = df['TEXT'].apply(normalize)

    df.to_csv('data/opinions_clean.csv', index=False, sep=';')