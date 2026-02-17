import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load datasets
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df1 = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
df2 = pd.read_csv(os.path.join(BASE_DIR, "ratings.csv"))



df = df2.merge(df1, on='movieId')
df = df[['userId', 'movieId', 'rating', 'title']]

# Create pivot table
user_movie_matrix = pd.pivot_table(df, values='rating', index='movieId', columns='userId')
user_movie_matrix = user_movie_matrix.fillna(0)

# User similarity matrix
user_user_matrix = user_movie_matrix.corr(method='pearson')

def recommend_movies(user_id):
    similar_users = user_user_matrix.loc[user_id].sort_values(ascending=False)[1:11]
    df_similar = pd.DataFrame(similar_users).reset_index()
    df_similar.columns = ['userId', 'similarity']

    final_df = df_similar.merge(df, on='userId')
    final_df['score'] = final_df['similarity'] * final_df['rating']

    watched = df[df['userId'] == user_id]['movieId']
    final_df = final_df[~final_df['movieId'].isin(watched)]

    recommendations = final_df.sort_values(by='score', ascending=False)['title'].head(10)
    return recommendations
st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=610, step=1)

if st.button("Get Recommendations"):
    results = recommend_movies(user_id)
    st.write("### Recommended Movies:")
    for movie in results:
        st.write(movie)


