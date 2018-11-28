
# coding: utf-8

# In[9]:

def presentation(product1):
    import pandas as pd
    import numpy as np
    import re
    data = pd.read_csv('products_YORCOM.csv', sep=',',encoding='latin-1')
    data=data.fillna("")
    def create_soup(x):
        return "".join(x['title'])  + "".join(x['name'])    + "".join(x['categorieen'])
    data['soup'] = data.apply(create_soup, axis=1)
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(ngram_range=(1,1))
    count_matrix = count.fit_transform(data["soup"])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    data = data.reset_index()
    indices = pd.Series(data.index, index=data['title'])
    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        product_indices = [i[0] for i in sim_scores]
        return data['title'].iloc[product_indices]
    return get_recommendations(product1, cosine_sim=cosine_sim) 


# In[10]:
Add something


