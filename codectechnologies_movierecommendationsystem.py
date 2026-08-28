import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Create a sample movie dataset
# ---------------------------------------------------------

movies = {
    "title": [
        "Avengers: Endgame",
        "Iron Man",
        "The Dark Knight",
        "Batman Begins",
        "Inception",
        "Interstellar",
        "Titanic",
        "The Notebook",
        "Toy Story",
        "Finding Nemo"
    ],

    "genres": [
        "Action Adventure Sci-Fi",
        "Action Adventure Sci-Fi",
        "Action Crime Drama",
        "Action Crime Drama",
        "Action Sci-Fi Thriller",
        "Adventure Drama Sci-Fi",
        "Drama Romance",
        "Drama Romance",
        "Animation Adventure Comedy",
        "Animation Adventure Family"
    ]
}

df = pd.DataFrame(movies)

print("Movie Dataset:")
print(df)


# ---------------------------------------------------------
# 2. Convert genres into numerical features
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

genre_matrix = vectorizer.fit_transform(df["genres"])


# ---------------------------------------------------------
# 3. Calculate similarity between movies
# ---------------------------------------------------------

similarity = cosine_similarity(genre_matrix)


# ---------------------------------------------------------
# 4. Create a recommendation function
# ---------------------------------------------------------

def recommend_movies(movie_title, number_of_recommendations=5):

    # Check whether movie exists
    if movie_title not in df["title"].values:
        print("Movie not found.")
        return

    # Find index of selected movie
    movie_index = df[df["title"] == movie_title].index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(similarity[movie_index]))

    # Sort movies by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself
    similarity_scores = similarity_scores[1:]

    print(f"\nMovies similar to '{movie_title}':")
    print("--------------------------------")

    # Display recommendations
    for index, score in similarity_scores[:number_of_recommendations]:
        print(
            f"{df.iloc[index]['title']} "
            f"(Similarity: {score:.2f})"
        )


# ---------------------------------------------------------
# 5. Test the recommendation system
# ---------------------------------------------------------

recommend_movies("Iron Man", 5)

recommend_movies("Titanic", 3)

recommend_movies("Toy Story", 3)
