import os
import pandas as pd
# import pipelines
from transformers import pipeline
import matplotlib.pyplot as plt

# path to "Fake or Real News" dataset
path_fake_real_news = os.path.join("in","fake_or_real_news.csv")

def get_data(data_path):
    # load data
    news_data = pd.read_csv(data_path, index_col=0) 
    # create list of only the headlines
    headlines = news_data["title"]
    print("Data is ready")

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
    print("Headlines are classified")

    return total_emotions, real_emotions, fake_emotions

def emotion_table(input_emotions, cathegory_name):
    # create table showin destribution of emotions in a category (real, fake or total respectively)
    input_emotion_table = pd.crosstab(index=input_emotions, columns="Count")
    # set outpath and name of table
    table_outpath = os.path.join("out", f"{cathegory_name}_table.csv")
    # save table as csv
    input_emotion_table.to_csv(table_outpath)
    print("Emotion tables are saved")

def save_plot(category_name):
    # set plot title
    plt.title(f"Distribution of emotions across {category_name}")
    # set outpath and name of plot
    outpath = os.path.join("out", f"{category_name}_bars.png")
    # save plot
    plt.savefig(outpath, dpi=400)
    # clear figure
    plt.clf() 

def plot_emotions(input_emotions, category_name):
    # define names of the seven emotions
    names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    # define emotions conters
    anger_count = 0
    disgust_count = 0
    fear_count = 0
    joy_count = 0
    neutral_count = 0
    sadness_count = 0
    surprise_count = 0
    # loop for counting each emotion type
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
    # add all emotion counts to a list
    values = [anger_count, disgust_count, fear_count, joy_count, neutral_count, sadness_count, surprise_count]
    # plot the values of the list
    plt.bar(names, values)
    # name of x-axis
    plt.xlabel("Emotion")
    # name of y-axis
    plt.ylabel("Count")
    # save bar plot
    save_plot(category_name)
    print("Bar chards are saved")

def main():
    #get headlines
    headlines, news_data = get_data(path_fake_real_news)
    # predict emotions in headlines
    total_emotions, real_emotions, fake_emotions = emotion_clf(headlines, news_data)

    # create and save table for emotions across ...
    # ...all headlines
    category_total = "total_emotions"
    emotion_table(total_emotions, category_total)
    # ... REAL headlines
    category_real = "real_emotions"
    emotion_table(real_emotions, category_real)
    # ... FAKE headlines
    category_fake = "fake_emotions"
    emotion_table(fake_emotions, category_fake)
    
    # create bar plots for all headlines, REAl headlines and FAKE headlines
    plot_emotions(total_emotions, category_total)
    plot_emotions(real_emotions, category_real)
    plot_emotions(fake_emotions, category_fake)

if __name__ == "__main__":
    main()