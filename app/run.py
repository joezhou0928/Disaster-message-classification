import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graph 1
    graph_1 = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    # graph 2
    df['num_cat'] = None
    df['num_cat'] = df.iloc[:,4:].sum(axis=1)
    num_cat = df.groupby('num_cat').count()['id'].index
    num_cat_count = df.groupby('num_cat').count()['id'].values
    
    graph_2 = {
            'data': [
                Bar(
                    x=num_cat,
                    y=num_cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Total Number of Related Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Total Number of Related Categories"
                }
            }
        }
    
    # graph 3
    category_names = df.drop(columns=['num_cat']).iloc[:,4:].sum(axis=0).sort_values(ascending=False).index
    category_frequency = df.drop(columns=['num_cat']).iloc[:,4:].sum(axis=0).sort_values(ascending=False).values
    
    graph_3 = {
            'data': [
                Bar(
                    x=category_names,
                    y=category_frequency
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Names"
                }
            }
        }      
        
       
    graphs = [graph_1,graph_2,graph_3]
    
    # add new plots here
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
