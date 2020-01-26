# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:36:56 2019

@author: modis1
"""
import pandas as pd
import gensim
import glob
file_names = glob.glob('/Users/modis1/Desktop/Python/document-similarity/JNK_EDITORIAL/*.xml')
file_names

dict_of_files = {}

for i in range(6136):
    print(i,file_names[i])
    dict_of_files.update({i:file_names[i].split(sep='\\')[1]})
    
    

for i in range(6136):
    print(i,file_names[i])
    
raw_documents=[]
for file in file_names:
    try:
        with open(file, "r", encoding="utf-8") as f : raw_documents.append(f.read())
    except:
        pass
    
print('number of documents %',len(raw_documents))

clean_texts = []
for text in raw_documents:
    clean_texts.append(gensim.utils.simple_preprocess(text))
    
clean_texts

dictionary1 = gensim.corpora.Dictionary(clean_texts)
print("Number of word in Dictionary:", len(dictionary1))

for i in range(100):
    print(i,dictionary1[i])
    
print(clean_texts[0])

corpus=[dictionary1.doc2bow(text) for text in clean_texts]

print(corpus[:10])

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

similarity_object = gensim.similarities.Similarity('/Users/modis1/Desktop/Python/document-similarity/JNK_EDITORIAL/', tf_idf[corpus], num_features=len(dictionary1))
print(similarity_object)
print(type(similarity_object))

list_result = []
for k,v in dict_of_files.items(): 
    text = raw_documents[k]
    query_doc = gensim.utils.simple_preprocess(text)
    print(query_doc)
    
    query_doc_bow=dictionary1.doc2bow(query_doc)
    print(query_doc_bow)
    
    query_doc_tf_idf = tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    
    similarity_scores=list(similarity_object[query_doc_tf_idf])
    similarity_scores
    
    max_score = max(similarity_scores)
    print(max_score)
    similarity_scores.index(max_score)
    
    print(raw_documents[k])
    
    sorted_scores=sorted(similarity_scores, reverse=True)
    print(sorted_scores[0])
    print(sorted_scores[1])
    
    index_number = similarity_scores.index(sorted_scores[1])
    
    print(raw_documents[index_number])
    
    list_result.append((dict_of_files[k],dict_of_files[index_number],sorted_scores[1]))


result=pd.DataFrame(list_result)
#result = result.transpose()
result.to_csv('/Users/modis1/Desktop/Python/document-similarity/OUT_METADATA_Mine/resultFinal{}.csv'.format("FullfinalResultforJNKbyTFIDF"),index=True)
    