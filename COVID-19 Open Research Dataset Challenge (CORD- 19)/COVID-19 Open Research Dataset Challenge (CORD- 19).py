#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.spatial
import numpy as np
import pandas as pd
import os, json
import glob
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import csv


# In[2]:


# with open ('../Big Data Project 2/trainning data.csv') as metedata:
# reader = csv.reader('../Big Data Project 2/trainning data.csv')
metadata = pd.read_csv('metadata.csv',
                  usecols=['cord_uid','source_x','title','license','publish_time',
                           'abstract','authors','journal','url'])


# In[3]:


file_path = ['pmc_json', 
             'pdf_json'
            ]

json_article_paths = []

for file in file_path:
    json_article_paths = json_article_paths + glob.glob(os.path.join(file, "*.json"))
    
print(len(json_article_paths))


# In[5]:


# a function to show high frequency words
def show_WordCloud(data):
    high_frequency_words = ' '
    stopwords = set(STOPWORDS) 

# iterate through the csv file 
    for word in data: 
        #change to string type
        word = str(word) 

        # split the value 
        tokens = word.split() 
        
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            high_frequency_words = high_frequency_words + words + ' '

    wordcloud = WordCloud(width = 500, height = 500, 
                    background_color ='white',
                    max_words = 200, 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(high_frequency_words) 

    # show result by plot                       
    plt.figure(figsize = (5, 5), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()


# In[12]:


i=0;

# synonyms to COVID-19 according to wikipedia
keywords = ['persistence','decontamination','RNA virus',' SARS','coronavirus', 'COVID', 'SARS-Cov-2', '-CoV', '2019-nCoV','coronavirus vaccine','Antibody-Dependent Enhancement','therapeutic','prophylaxis clinical','naproxen','clarithromycin','minocyclinethat']

titles = []
paper_ids = []
paper_url = []
abstracts = []


for json_file in json_article_paths: 
    
    # read json file into doc, append article only if it contains any of the keywords in its title
    # limit the number of article for running speed
    if i<5000:
        document = json.load(open(json_file)) 
        title = document['metadata']['title']  
        title = re.sub(r'[^\x00-\x7F]',' ', title)

        if title != '' and any(keyword.lower() in title.lower() for keyword in keywords):
         
        #             abstract = metadata['abstract']
            titles.append(title)
            paper_ids.append(document['paper_id'])
#             abstracts.append(abstract)
            
    i+=1

print(len(paper_ids))


# In[13]:


keyword_articles_df = pd.DataFrame({
    'title': titles, 
    'paper_id': paper_ids
})

# pd.read_csv('metadata.csv')

filtered_data = pd.merge(metadata, keyword_articles_df)
filtered_data = filtered_data.drop_duplicates(subset='title')
filtered_data = filtered_data.dropna(subset=['abstract'])
    
filtered_data.head()


# In[8]:


show_WordCloud(abstracts)


# In[14]:


# function
def get_top_similar_articles(query, content):
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    query_embeddings = embedder.encode([query])

    # list of article titles
    content_embeddings = embedder.encode(content)

    # get top article titles based on cosine similarity
    closest_n = 10
    
    distances = scipy.spatial.distance.cdist(query_embeddings, content_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    # save similar articles info
    top_titles = []
    top_abstracts = []
    top_paper_ids = []
    top_similarity_scores = []
    
    abstracts = list(filtered_data.abstract)

    print('Query: ' + query + '\n')

    # Find the closest article titles for each query sentence based on cosine similarity
    for idx, distance in results[0:closest_n]:
        top_paper_ids.append(paper_ids[idx])
        top_titles.append(titles[idx])
        top_similarity_scores.append(round((1-distance), 4))
        top_abstracts.append(abstracts[idx])
        print('Paper ID: ' + paper_ids[idx])
        print('PubMed Article Title: ' + titles[idx])
        print('Similarity Score: ' + "%.4f" % (1-distance))
        print('\n')
        
    top_similar_articles_df = pd.DataFrame({
        'paper_id': top_paper_ids,
        'cosine_similarity': top_similarity_scores,
        'title': top_titles,
        'abstract': top_abstracts
    })
    
    return top_similar_articles_df


# In[15]:


# research question
queries = ['What is the efficacy of novel therapeutics being tested currently?', 
           'What is the best method to combat the hypercoagulable state seen in COVID-19?']

q1_top_similar_articles_df = get_top_similar_articles(queries[0], titles)


# In[ ]:




