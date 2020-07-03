# Disaster Response Pipeline Project
### Overview:
This project is about disaster messages classification based on machine learning. During each disaster event, people would post messages in different categories depending on their urgent needs. Knowing the category of each message can improve the efficiency of disaster relief agencies' work. An web app is also built to visualize the analysis.

### Instructions:
1. Run the following commands to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database (needed files are under the folder called data)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves (needed file is under the folder called models)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://SPACEID-3001.SPACEDOMAIN to check your application! To know the SPACEID and SPACEDOMAIN, type in 'env | grep WORK' in your command prompt.
