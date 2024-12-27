import pandas as pd
import requests
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
data = pd.read_csv("recipes.csv")

# Function to fetch images from Pexels API
def fetch_image(recipe_name, api_key):
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": recipe_name, "per_page": 1}  # Fetch only one image per recipe
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        result = response.json()
        if result["photos"]:
            return result["photos"][0]["src"]["medium"]  # Get the medium-sized image URL
        else:
            return "https://via.placeholder.com/150"  # Placeholder if no image is found
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "https://via.placeholder.com/150"  # Placeholder for errors

# API Key (replace 'YOUR_API_KEY' with the key from Pexels)
PEXELS_API_KEY = "i0ce1iNvt1kZoDNrBBwZCw7wPLbNuzQwZmqEwzUV5KlLnJVkc3DQVysC"

# Tokenize ingredients
data['ingredient_tokens'] = data['ingredients'].apply(lambda x: x.split(", "))
data['ingredient_str'] = data['ingredients'].apply(lambda x: x.lower())  # For matching

# Fetch images for each recipe
data['photo_url'] = data['name'].apply(lambda x: fetch_image(x, PEXELS_API_KEY))

# Create TF-IDF vectorizer and fit it on the recipes
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['ingredients'])

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data['ingredient_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save models and data
with open("recipe_model.pkl", "wb") as f:
    pickle.dump((tfidf_matrix, vectorizer, data), f)

word2vec_model.save("word2vec.model")

# Save tokenized data and model
pickle.dump((data, word2vec_model), open("recipe_model_word2vec.pkl", "wb"))

print("Model training and saving completed!")
