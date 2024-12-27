from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pre-trained model and data
tfidf_matrix, vectorizer, recipes = pickle.load(open("recipe_model.pkl", "rb"))
data, word2vec_model = pickle.load(open("recipe_model_word2vec.pkl", "rb"))

def compute_similarity(user_ingredients, recipe_ingredients):
    user_valid_words = [word for word in user_ingredients if word in word2vec_model.wv]
    recipe_valid_words = [word for word in recipe_ingredients if word in word2vec_model.wv]

    if not user_valid_words or not recipe_valid_words:
        return 0.0

    user_vec = np.mean([word2vec_model.wv[word] for word in user_valid_words], axis=0)
    recipe_vec = np.mean([word2vec_model.wv[word] for word in recipe_valid_words], axis=0)

    if user_vec is None or recipe_vec is None or len(user_vec) == 0 or len(recipe_vec) == 0:
        return 0.0

    return cosine_similarity([user_vec], [recipe_vec])[0][0]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['ingredients']
    user_ingredients = set(user_input.lower().replace(" ", "").split(","))

    recipes['contains_ingredient'] = recipes['ingredient_tokens'].apply(
        lambda x: any(ingredient in x for ingredient in user_ingredients)
    )

    top_recipes = recipes[recipes['contains_ingredient']].copy()
    top_recipes['similarity_score'] = top_recipes['ingredient_tokens'].apply(
        lambda x: compute_similarity(user_ingredients, x)
    )

    top_recipes = top_recipes.sort_values(by='similarity_score', ascending=False).head(5)

    return render_template("results.html", recipes=top_recipes.to_dict(orient="records"))

@app.route('/recipe/<int:recipe_id>', methods=['GET'])
def recipe_detail(recipe_id):
    recipe = recipes[recipes['id'] == recipe_id]
    if recipe.empty:
        return "Recipe not found", 404

    recipe = recipe.iloc[0]
    return render_template("recipe_detail.html", recipe=recipe)

if __name__ == "__main__":
    app.run(debug=True)

#hello