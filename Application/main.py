# Author: Garth Scheck
# Date: 2/22/2025
# Summary: This is a Python application for recommending movies.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import requests
import json
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


# gets data from API and stores in CSV file
def get_movie_data(start):
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxZWYzZmU5N2RjMjY2Nzc0YmRlMWRhOGI3MjNiZGNlNSIsIm5iZiI6MTczODM0NDgxNi41Mzc5OTk5LCJzdWIiOiI2NzlkMDk3MDNkZTU4OGQ1YTAwMTMzOGEiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.s-Z46xcihvHIW5OFuIDSqKD6qTZXcLoAveYeNZ-aRIY"
    }

    df = pd.DataFrame(columns=['id', 'user_id', 'title', 'release_dt', 'genre', 'vote', 'popularity', 'liked'])

    movie_id = 1

    end = start+5

    for page in range(start, end):
        url = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=" + str(
            page) + "&sort_by=popularity.desc"
        response = requests.get(url, headers=headers)

        data = response.json()

        for result in data['results']:
            for genre in result['genre_ids']:
                igenre = int(genre)
                df.loc[len(df.index)] = [movie_id, 1, result['original_title'], result['release_date'], igenre,
                                         result['vote_average'], result['popularity'], 0]

            movie_id += 1
 
        sleep(2)

    # uncomment the following line to save data to csv file
    # line has been commented out because labeling of data has not been implmented yet
    # df.to_csv('c:/temp/movie_data.csv', index=False)
    return df


# gets recommendations from a new data set
def get_recommendations(page_num):
    df = pd.read_csv("c:/temp/movie_data.csv")
    cdf = df[['id', 'user_id', 'release_dt', 'genre', 'vote', 'popularity', 'liked']].copy()

    cdf['release_dt'] = pd.to_datetime(cdf['release_dt'])

    cdf['day'] = cdf['release_dt'].dt.day
    cdf['month'] = cdf['release_dt'].dt.month
    cdf['year'] = cdf['release_dt'].dt.year

    cdf.drop('release_dt', axis=1, inplace=True)

    X = cdf[['day', 'month', 'year', 'genre', 'vote', 'popularity']]  # Features
    y = cdf['liked']  # Target

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train Naive Bayes model
    model = GaussianNB()
    # model = MultinomialNB()
    model.fit(X_train, y_train)

    # Step 6: Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Recommend movies with highest probability of being liked
    page_num = int(page_num)
    df = get_movie_data(page_num)
    cdf = df[['id', 'user_id', 'release_dt', 'genre', 'vote', 'popularity', 'liked']].copy()

    cdf['release_dt'] = pd.to_datetime(cdf['release_dt'])

    cdf['day'] = cdf['release_dt'].dt.day
    cdf['month'] = cdf['release_dt'].dt.month
    cdf['year'] = cdf['release_dt'].dt.year

    cdf.drop('release_dt', axis=1, inplace=True)

    X = cdf[['day', 'month', 'year', 'genre', 'vote', 'popularity']]

    df['probability_like'] = model.predict_proba(X)[:, 1]
    df = df[(df['probability_like'] < 0.999) & (df['probability_like'] > 0.8)]

    # Sort and drop duplicates
    recommendations = df.sort_values(by='probability_like', ascending=False)

    recommendations = recommendations.drop_duplicates(subset='id', keep="last")
    return recommendations


# main starting point of application
if __name__ == '__main__':
    page = input("Enter movie page: ")

    if page.isnumeric():
        rec = get_recommendations(page)

        print("\nTop Recommendations:")
        print(rec[['title', 'probability_like']])
    else:
        print("Exiting application.")

