import os

import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


def read_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the datasets.
    Returns:
        tuple containing DataFrames for books and ratings
    """
    # books = pd.read_csv("rec_sys/dataset/Books.csv", low_memory=False)
    # ratings = pd.read_csv("rec_sys/dataset/Ratings.csv")
    books = pd.read_csv("C:/Users/Heshan_2/PycharmProjects/book-recommendation-system_using_h2o-wave/rec_sys/dataset/movies.csv", low_memory=False)
    ratings = pd.read_csv("C:/Users/Heshan_2/PycharmProjects/book-recommendation-system_using_h2o-wave/rec_sys/dataset/Movie_ratings.csv")

    return books, ratings


# def save_data(rating_matrix, similarity_scores, books):
#     """
#     Saves rating matrix, books, book names, and similarity scores in pickle format.
#     Parameters:
#         rating_matrix: pandas.DataFrame - User rating for each book
#         similarity_scores:
#         books: pandas.DataFrame - Details of the books
#     """
#     pickle.dump(
#         list(rating_matrix.index), open("rec_sys/rec_data/book_names.pkl", "wb")
#     )
#     pickle.dump(rating_matrix, open("rec_sys/rec_data/books.pkl", "wb"))
#     pickle.dump(books, open("rec_sys/rec_data/books.pkl", "wb"))
#     pickle.dump(similarity_scores, open("rec_sys/rec_data/similarity_scores.pkl", "wb"))

def save_data(rating_matrix, similarity_scores, books):
    """
    Saves rating matrix, books, book names, and similarity scores in pickle format.
    Parameters:
        rating_matrix: pandas.DataFrame - User rating for each book
        similarity_scores:
        books: pandas.DataFrame - Details of the books
    """
    save_path = "rec_sys/rec_data/"

    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save movie names
    movie_names_path = os.path.join(save_path, "movie_names.pkl")
    if not os.path.exists(movie_names_path):
        pickle.dump(list(rating_matrix.index), open(movie_names_path, "wb"))

    # Save movie rating matrix
    rating_matrix_path = os.path.join(save_path, "movie_rating_matrix.pkl")
    if not os.path.exists(rating_matrix_path):
        pickle.dump(rating_matrix, open(rating_matrix_path, "wb"))

    # Save movies
    movies_path = os.path.join(save_path, "movies.pkl")
    if not os.path.exists(movies_path):
        pickle.dump(books, open(movies_path, "wb"))

    # Save similarity scores
    similarity_scores_path = os.path.join(save_path, "similarity_scores_movies.pkl")
    if not os.path.exists(similarity_scores_path):
        pickle.dump(similarity_scores, open(similarity_scores_path, "wb"))


def rec_init():
    """
    Computes the similarity scores based on collaborative filtering.
    Users that reviewed more than 200 books and books with equal or more than 50 ratings 
    are considered to improve the quality of recommendations. Similarity is measured based
    on cosine-similarity.
    """
    books, ratings = read_dataset()

    ratings_with_books = ratings.merge(books, on="Movie-ID")

    # Finding users with more than 200 reviews.
    ratings_group = ratings_with_books.groupby("User-ID").count()["Movie-Rating"]
    ratings_group = ratings_group[ratings_group > 200]

    ratings_filtered = ratings_with_books[
        ratings_with_books["User-ID"].isin(ratings_group.index)
    ]

    # Finding books with equal or more than 50 ratings.
    filtered_group = ratings_filtered.groupby("Movie-Title").count()["Movie-Rating"]
    filtered_group = filtered_group[filtered_group >= 50]

    final_filtered_ratings = ratings_filtered[
        ratings_filtered["Movie-Title"].isin(filtered_group.index)
    ]

    rating_matrix = final_filtered_ratings.pivot_table(
        index="Movie-Title", columns="User-ID", values="Movie-Rating"
    )
    rating_matrix.fillna(0, inplace=True)

    similarity_scores = cosine_similarity(rating_matrix)
    save_data(rating_matrix, similarity_scores, books)


if __name__ == "__main__":
    rec_init()
