import os
import pandas as pd
# import pipelines
from transformers import pipeline
import random2 as rd
import matplotlib.pyplot as plt

def get_data():
    # define path
    data_path = os.path.join("in","fake_or_real_news.csv")
    # load data
    news_data = pd.read_csv(data_path, index_col=0) 
    # create list of only the headlines
    headlines = news_data["title"]
    # create list of only FAKE headlines
    fake_headlines = []
    for label in news_data: 
        if label == "FAKE":
            #if this, then grap the "title" from the same row
            fake_headlines.append(news_data["title"])
        
    # create list of only REAL headlines

    #sampling random headliens --> REMOVE BEFORE HAND IN
    headlines = rd.sample(headlines, 10)

    return headlines, fake_headlines

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
    
def plot_emotions(labels):
    names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    values=[labels.count("anger"),
            labels.count("disgust"), 
            labels.count("fear"), 
            labels.count("joy"), 
            labels.count("neutral"), 
            labels.count("sadness"),
            labels.count("surprise")]
    bar_plot = plt.bar(names, values)
    plt.suptitle('Destribution of emotions across all headlines')
    return bar_plot

def main():
    #get headlines
    headlines, fake_headlines = get_data()
    print("Headlines ready!")
    print(fake_headlines)
    # predict emotions in headlines
    labels = emotion_clf(headlines)
    print("Labels for headlines predicted")
    # creating the bar plot
    bar_plot = plot_emotions(labels)
    plt.savefig('out/all_headliens.png',dpi=400)
    
    ### Create matrix showin results 
    #dataframes = []
    #dataframe = pd.DataFrame(labels, columns=["Headlines", "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
    #dataframes.append(dataframe)
    #emotion_data = pd.concat(dataframes)

    #save dataframe as csv
    #folderdata.to_csv(f'out/labels.csv', index=False)

    print(f"Dataframe is saved")

if __name__ == "__main__":
    main()