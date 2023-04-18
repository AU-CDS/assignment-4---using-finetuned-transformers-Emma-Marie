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

    return headlines, news_data

def emotion_clf(headlines, news_data):
    # load emotion pipeline from Hugging Face
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=False) # return only the highest score of each headline
    # run the classifier on the headlines data and extract only the label names
    emotions = []
    for headline in headlines:
        emotions.append(classifier(headline)[0]["label"])
    # create data frame for all headlines
    total_emotions = pd.Series(emotions, index=news_data.index, name="Emotion")
    # data frame for emotions in FAKE headlines
    real_news_mask = news_data["label"]=="REAL"
    real_emotions = total_emotions.loc[real_news_mask].reset_index(drop=True).rename("Emotion")
    # data frame for emotions in READL headlines
    fake_news_mask = news_data["label"]=="FAKE"
    fake_emotions = total_emotions.loc[fake_news_mask].reset_index(drop=True).rename("Emotion")

    return total_emotions, real_emotions, fake_emotions

def plot_emotions(input_emotions):
    names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    anger_count = 0
    disgust_count = 0
    fear_count = 0
    joy_count = 0
    neutral_count = 0
    sadness_count = 0
    surprise_count = 0
    for emotion in input_emotions:
        if emotion == "anger":
            anger_count+=1
        elif emotion =="disgust":
            disgust_count+=1
        elif emotion =="fear":
            fear_count+=1
        elif emotion =="joy":
            joy_count+=1
        elif emotion =="neutral":
            neutral_count+=1
        elif emotion =="sadness":
            sadness_count+=1
        elif emotion =="surprise":
            surprise_count+=1
        else: 
            pass
    values = [anger_count, disgust_count, fear_count, joy_count, neutral_count, sadness_count, surprise_count]
    plt.bar(names, values)
    plt.xlabel("Emotion")
    plt.ylabel("Count")

def main():
    #get headlines
    headlines, news_data = get_data()
    # predict emotions in headlines
    total_emotions, real_emotions, fake_emotions = emotion_clf(headlines, news_data)

    # create and save table for emotions across all headlines
    all_headlines_table = pd.crosstab(index=total_emotions, columns="Count")
    all_headlines_table.to_csv("out/all_headlines_table.csv")
    # table for emotions across REAL headlines
    real_headlines_table = pd.crosstab(index=real_emotions, columns="Count")
    real_headlines_table.to_csv("out/real_headlines_table.csv")
    # table for emotions across FAKE headlines
    fake_headlines_table = pd.crosstab(index=fake_emotions, columns="Count")
    fake_headlines_table.to_csv("out/fake_headlines_table.csv")
    
    # plot_emotions for all headlines
    plot_emotions(total_emotions)
    plt.title(f"Distribution of emotions across all headlines")
    plt.savefig(f"out/total_headlines_bars.png",dpi=400)
    plt.clf() # clear figure
    # plot for REAl headlines
    plot_emotions(real_emotions)
    plt.title(f"Distribution of emotions across real headlines")
    plt.savefig(f"out/real_headlines_bars.png",dpi=400)
    plt.clf() # clear figure
    # plot for FAKE headlines
    plot_emotions(fake_emotions)
    plt.title(f"Distribution of emotions across fake headlines")
    plt.savefig(f"out/fake_headlines_bars.png",dpi=400)

if __name__ == "__main__":
    main()