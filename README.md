[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10838296&assignment_repo_type=AssignmentRepo)
# Assignment 4 - Using finetuned transformers via HuggingFace

## Purpose
In this assignment, I will use HuggingFace to do feature extraction from the ```Fake or Real News``` dataset. I use the ```Emotion English DistilRoBERTa-base```model to classify the emotions in the news headlines in the data set. The model predicts six basic emotions and one neutral emotion. The emotions are: anger, disgust, fear, joy, neutral, sadness, and surprise. The results are displaied in three tables and three bar plots,  which can be found in the ```out``` folder.

Further documentation for the model from HuggingFace can be found here: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base 

## Scripts
This project contains one script called ```emotion_clf```, which can be found in the ```src```folder. The script trains the emotion classifier on the headlines from the Fake or Real News data set. The script consists of the following parts:

Functions: 
- __get_data()__ loads in the data and extracts the news headlines. 
- __emotion_clf()__ loads the emotionclassifier model, runs the classifier on the headlines, and create three data frames containing the distribution of emotions across three categories: 1) _all_ headlines, 2) headlines from _real_ news, 3) headlines from _fake_ news.
- __plot_emotions()__ counts in how many headlines each emotion occurs, and creates a bar chart showing the desctribution of emotions. Is run on each of the three data frames creates in the emotion_clf(). 
- __main()__ runs the three functions above. Furthermore, it  creates three tables each for each of the three categories (all headlines, real headlines, fake headlines) to display the results. The tables show the desctribution of emotions across the headlines in the given category.


## Data
The data is used to train the model in this assignemtn is called```Fake or Real News```. The data can be found in the ```in```folder, but is originally from Kaggle. The data consists of a lot of different news articles of which half os _fake news_ and the other half is _real news_. The data set is an array with three columns: "title", "text", and "label". For this assignment, I'm only interested in the titles and their label (REAL or FAKE). 

The data can be accessed and download (12 MB) here: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news. Note that the data has the additional column "number" (compared to the version in the "in" folder) when downloaded from Kaggle. 

## How to run the scripts
Run the emotion_clf.py:
- run "bash setup.sh" from the commandline to create a virtual environment and install the required packages.
- run "bash run.sh" from the commandline to activate the virtual environment, run the code, and deactivate the environment.

## Discussing of the results

The majority of the headlines in the total data set are neutral. The second most common emotion is fear, and the third is anger. Sadness is number four and only occurs 53 times more than disgust. Suprise is number six and joy is number seven. This means, that negative emotions dominate the total corpus of news articles. Only 363 of the 6335 headlines belongs to one of the two possitive emitions _joy_ and _surprise_. 

The destribution of emotions across the real headlines and the fake headlines looks very similar. They also show the same tendencies as data set of all the headlines: most headlines are neutral, and only few express joy or surprise. 
I expected the proportion of negative emotions among fake news headlines to be bigger than the proportion of negative emotions among real news headlines, because I was under the (not accademically substantiated) impression, that fake news are often used to spread negative rumours about other people or events. But according to this model, the negative emotions are not a special character trait for fake news. 

## References
Fake or Real News, Kaggle: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news 

Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
