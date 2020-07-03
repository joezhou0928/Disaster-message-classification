# import the needed packages
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    """
    Args:
    database_filepath: file path to the database from which we read data
    
    Return:
    X: all messages (input)
    Y: all categories (labels)
    category_name: a list of category names
    """    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    return X,Y,category_names

def tokenize(text):
    """
    Args:
    text: sequence of words
    
    Return:
    clean_tokens: tokens after processing 
    """        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(tok)
    return clean_tokens

def build_model():
    """
    Args: None
        
    Return:
    cv: a trained model after grid search cross-validation
    """
    """create a transformer named TextLengthExtractor
       that can extract text length as features
    """
    # set up a machine learning pipeline 
    pipeline = Pipeline([
            ('vec',CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    # set up parameters for grid search
    # I only put one value here to save execution time
    # because finding the best parameters is probably not the focus here
    parameters = {
        'clf__estimator__min_samples_split': [3]     
    }
    # run grid search
    cv = GridSearchCV(pipeline,param_grid = parameters)
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args: 
    model: a trained machine learning model
    X_test: x in the test set
    Y_test: y in the test set
    category_names: a list of category names
        
    Return: None
    """
    # make predictions
    Y_pred = model.predict(X_test)
    # for each category, print the classification report
    for i in range(Y_pred.shape[1]):
        print(category_names[i],classification_report(Y_test.iloc[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
