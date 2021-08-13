import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

columns_names = ["user_id","item_id","rating","timestamp"]
df=pd.read_csv("100kDS/u.data",sep='\t',names=columns_names)

movie_titles=pd.read_csv("100kDS/u.item", sep="\|", header=None)

movie_title=movie_titles[[0,1]]

movie_title.columns= ['item_id','title']

df=pd.merge(df,movie_title,on="item_id")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

df.groupby('title').mean()

df.groupby('title').count()['rating'].sort_values(ascending=False)

ratings=pd.DataFrame(df.groupby('title').mean()['rating'])

ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])

ratings.sort_values(by='rating',ascending=False)

#plt.figure(figsize=(10,6))
#plt.hist(ratings['num of ratings'] , bins=70)

#plt.hist(ratings['rating'],bins=70)

#sns.jointplot(x='rating',y='num of ratings', data=ratings, alpha=0.5)

moviemat = df.pivot_table(index="user_id", columns="title",values="rating")

ratings.sort_values('num of ratings',ascending=False).head()

def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    
    return predictions

predictions = predict_movies("Titanic (1997)")

predictions.head()    
    