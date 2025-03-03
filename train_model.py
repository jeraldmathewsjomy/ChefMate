import pandas as pd
import requests
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
data = pd.read_csv("recipes.csv")

# Function to fetch images from Pexels API
def fetch_image(recipe_name, api_key):
    if not isinstance(recipe_name, str) or not recipe_name.strip():
        return "https://via.placeholder.com/150"  # Default image for empty names

    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": recipe_name, "per_page": 1}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        result = response.json()
        if "photos" in result and result["photos"]:
            return result["photos"][0]["src"]["medium"]
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")

    return "https://via.placeholder.com/150"  # Default placeholder image

# Replace with your actual API key
PEXELS_API_KEY = "i0ce1iNvt1kZoDNrBBwZCw7wPLbNuzQwZmqEwzUV5KlLnJVkc3DQVysC"

# Ensure 'id' column is correctly formatted
data['id'] = pd.to_numeric(data['id'], errors='coerce').fillna(-1).astype(int)

# Handle NaN values in ingredients
data['ingredients'] = data['ingredients'].astype(str).fillna("")

# Tokenize ingredients safely
data['ingredient_tokens'] = data['ingredients'].apply(lambda x: x.split(", ") if isinstance(x, str) else ["unknown"])

# Convert to lowercase
data['ingredient_str'] = data['ingredients'].str.lower()

# Fetch images for each recipe
data['photo_url'] = data['name'].apply(lambda x: fetch_image(x, PEXELS_API_KEY))

# Ensure cooking_steps are properly formatted
data['cooking_steps'] = data['cooking_steps'].fillna("No cooking steps available.").astype(str)

# Convert pipe-separated steps into a list
data['cooking_steps'] = data['cooking_steps'].apply(lambda x: x.split("|") if "|" in x else [x])


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['ingredients'])

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data['ingredient_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save models and data
with open("recipe_model.pkl", "wb") as f:
    pickle.dump((tfidf_matrix, vectorizer, data), f)

word2vec_model.save("word2vec.model")

pickle.dump((data, word2vec_model), open("recipe_model_word2vec.pkl", "wb"))

print("Model training and saving completed successfully!")
