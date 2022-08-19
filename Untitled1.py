#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd 


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[4]:


movies.head(2)


# In[5]:


movies.shape


# In[6]:


credits.head()


# In[19]:


#movies = movies.merge(credits,on='title')
movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[20]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# movies.info()

# In[23]:


movies.isnull().sum()


# In[24]:


movies.dropna(inplace=True)


# In[25]:


movies.iloc[0].genres


# In[27]:


import ast


# In[28]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[30]:


movies.dropna(inplace=True)


# In[31]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[32]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[35]:


movies['cast_x'][0]


# In[37]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[39]:


movies['cast_x'] = movies['cast_x'].apply(convert)
movies.head()


# In[40]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[42]:


movies['crew'] = movies['crew_x'].apply(fetch_director)


# In[43]:


movies.sample(5)


# In[44]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[45]:


movies.head()


# In[47]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast_x']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew_x']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])



# movies.head()

# In[48]:


movies.head()


# In[50]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast_x']+movies['crew_x']


# In[51]:


movies.head()


# In[54]:


new_df= movies[['movie_id_x','title','tags']]


# In[56]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[57]:


new_df.head()


# In[58]:


new_df['tags'][0]


# In[59]:


new_df['tags'].apply(lambda x: x.lower())


# In[60]:


new_df.head()


# In[75]:


#Vectorisation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[76]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[77]:


vector


# In[79]:


cv.get_feature_names()


# In[68]:


#Stemming-> ['Loving','loved','Love'] as input -> Output as ['Love','Love','Love']


# In[70]:


import nltk


# In[71]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[72]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[74]:


new_df['tags']=new_df['tags'].apply(stem)


# In[80]:


#Use cosine similarity for calculating distance to avoid of curse of dimeensionality
from sklearn.metrics.pairwise import cosine_similarity


# In[82]:


cosine_similarity(vector)


# In[85]:


similarity=cosine_similarity(vector)


# In[111]:


similarity
        


# In[90]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[121]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(similarity[movie_index])),reverse=True,key = lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[122]:


recommend('Avatar')


# In[120]:


new_df.iloc[1216].title

