import os
import pandas as pd
# import pipelines
from transformers import pipeline
import random2 as rd

def get_data():
    # define path
    data_path = os.path.join("in","fake_or_real_news.csv")
    # load data
    news_data = pd.read_csv(data_path, index_col=0) 
    # create list of only the headliens
    headlines = news_data["title"]
    #sampling random headliens --> REMOVE BEFORE HAND IN
    headlines = rd.sample(headlines, 10)
    #headlines = news_data["title"]

    return headlines

def emotion_clf(headlines):
    # load emotion pipeline from Hugging Face
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=False) # the "False" gives only the highest score of each document
    labels = []
    # run the classifier on the headlines data and extract only the label names
    for headline in headlines:
        labels.append(classifier(headline)[0]["label"])
    
    return labels
    

def main():
    #get headlines
    headlines = get_data()
    print("Headlines ready!")
    # predict emotions in headlines
    labels = emotion_clf(headlines)
    print("Labels for headlines predicted")
    # Create matrix showin results 
    dataframes = []
    dataframe = pd.DataFrame(labels, columns=["Headlines", "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
    dataframes.append(dataframe)
    emotion_data = pd.concat(dataframes)

    #save dataframe as csv
    folderdata.to_csv(f'out/labels.csv', index=False)

    print(f"Dataframe is saved")

if __name__ == "__main__":
    main()