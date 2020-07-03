# import packages needed
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Args:
    messages_filepath: filepath to the csv file with message data
    categories_filepath: filepath to the csv file with categories data
    
    Return:
    df: A dataframe that merges two data files on the 'id' column
    """
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = ['id'])
    return df
def clean_data(df):
    """
    Args:
    df: A dataframe needs to be cleaned
    
    Return:
    df: A CLEAN dataframe
    """
    """ Split categories into columns; Rename columns with category names"""
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = categories.iloc[0,:].apply(lambda x:x[:-2]).values
    categories.columns = category_colnames
    """ Convert each column to binary values"""
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    """ Drop the original category column in the df; Concat df with the new category column"""
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True) #remove all duplicates
    return df
def save_data(df, database_filename):
    """
    Args:
    df: A dataframe needs to be stored
    database_filename: The name of the database
    
    Return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('data', engine, index=False)  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
