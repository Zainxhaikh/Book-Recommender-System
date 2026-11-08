from flask import Flask, render_template, request
import pickle
import numpy as np

popular_df = pickle.load(open("popular.pkl", "rb"))
pt = pickle.load(open("pt.pkl", "rb"))
books = pickle.load(open("books.pkl", "rb"))
similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",
                           book_name=list(popular_df["Book-Title"].values)[:50],
                           author=list(popular_df["Book-Author"].values)[:50],
                           image=list(popular_df["Image-URL-M"].values)[:50],
                           votes=list(popular_df["num-ratings"].values)[:50],
                           ratings=list(popular_df["avg-rating"].values)[:50],
                           )

@app.route("/recommend")
def recommend_ui():
    return render_template("recommend.html")

@app.route("/recommend_books", methods=["post"])
def recommend_books():
    user_input = request.form.get("user_input")
    # index fetch
    index = np.where(pt.index == user_input)[0][0]

    distances, indices = similarity_scores.kneighbors(pt.iloc[index].values.reshape(1, -1), n_neighbors=6)

    data = []
    for i in indices[0][1:]:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return render_template("recommend.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
    