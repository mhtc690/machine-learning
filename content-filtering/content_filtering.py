#!/Users/mahe/anaconda3/bin/python
# content_filtering.py

import numpy as np
import pandas as pd

data_movies = pd.read_csv('movies.csv')
data_rating = pd.read_csv('ratings.csv')

# process 'data_movies' structure
# 将movies 'titles'里面的年份取出来
# 将genres展开成数组形势
data_movies['year'] = data_movies['title'].str.extract(r'(\d\d\d\d)',
                                                       expand=False)
data_movies['title'] = data_movies['title'].str.replace(r'(\(\d\d\d\d\))', '')
data_movies['title'] = data_movies['title'].apply(lambda x: x.strip())
data_movies['genres'] = data_movies['genres'].str.split('|')

# copy a data structure and use one-hot encoding for genres(题材)
data_movies_copy = data_movies.copy()
for index, row in data_movies.iterrows():
    for genre in row['genres']:
        data_movies_copy.at[index, genre] = 1
data_movies_copy = data_movies_copy.fillna(0)

# process 'data_rating' structure
data_rating = data_rating.drop('timestamp', 1)

# create a user input
userInput = [
            {'title': 'Breakfast Club, The', 'rating': 5},
            {'title': 'Toy Story', 'rating': 3.5},
            {'title': 'Jumanji', 'rating': 2},
            {'title': "Pulp Fiction", 'rating': 5},
            {'title': 'Akira', 'rating': 4.5}
         ]
userInput = pd.DataFrame(userInput)
# 找到data_movies中和userInput有同样‘title’的内容，合并到userInput上
inputId = data_movies[data_movies['title'].isin(userInput['title'].tolist())]
userInput = pd.merge(inputId, userInput)
userInput = userInput.drop('genres', 1).drop('year', 1)

# 和刚刚同样的寻找过程，用movieId更节省计算资源
userInput_en = data_movies_copy[data_movies_copy['movieId'].isin(
        userInput['movieId'].tolist())]
userInput_en = userInput_en.reset_index(drop=True)
userInput_en = userInput_en.drop('movieId', 1).drop('title', 1).drop(
        'genres', 1).drop('year', 1)

# start to calculate weights
userProfile = userInput_en.transpose().dot(userInput['rating'])
data_movies_copy = data_movies_copy.drop('movieId', 1)\
                   .drop('title', 1)\
                   .drop('genres', 1)\
                   .drop('year', 1)

# Calculate recommend table
# recommend_table = data_movies_copy.dot(userProfile)/(userProfile.sum())
recommend_table = (data_movies_copy*userProfile).sum(axis=1)\
                    / userProfile.sum()
recommend_table = recommend_table.sort_values(ascending=False)
recommend_table = data_movies[
    data_movies['movieId'].isin(recommend_table.head(20).keys())
]
print(recommend_table.head())
