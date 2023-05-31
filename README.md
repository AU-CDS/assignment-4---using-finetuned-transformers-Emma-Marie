[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10838296&assignment_repo_type=AssignmentRepo)
# Assignment 4 - Using finetuned transformers via HuggingFace

## 1. Contribution
I have developed the code for this assignment without other contributors.

## 2. Description
In this assignment, I will use ```HuggingFace``` to do feature extraction from the ```Fake or Real News``` dataset. I use the ```Emotion English DistilRoBERTa-base``` model to classify the emotions in the news headlines in the data set. The model predicts six basic emotions and one neutral emotion. The emotions are anger, disgust, fear, joy, neutral, sadness, and surprise. The results are displayed in the three tables and the three bar plots, which can be found in the ```out``` folder. Further documentation of the model from HuggingFace can be found here: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base. 

## 3. Methods
This assignment consists of one script called ```emotions_clf.py``` which is in the ```src``` folder. The data is loaded and the hews headlines are extracted. The emotion classifier model is loaded and run on the headlines. The output is three data frames showing the distribution of emotions across three categories: *all* headlines, headlines from *real* news, and headlines from *fake* news. For each of the three categories of news headlines, the number of headlines showing each emotion are counted, and the results are shown in a bar chart for each category. Chards and data frames are saved in the ```out``` folder.

## 4. Data
The data for this assignment is called Fake or Real News. The data consists of a lot of different newspaper articles of which half is fake news and the other half is real news. The data set is an array with three columns: “title”, “text”, and “label”. For this assignment, I’m only interested in the titles and their labels. The two labels are “FAKE” and “REAL”. 

### 4.1 Get the data
The data can be found in the ```in``` folder. If needed, it can also be downloaded (12 MB) from Kaggle: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news. Note that the data has the additional column “number” when downloaded from Kaggle, and this column is not part of the version of the dataset provided by my teacher Ross, which is in the one in the GitHub repository.

## 5. Usage
### 5.1 Prerequisites
For the scripts to run properly, please install Python 3 and Bash. The code for this assignment is created and tested using the app “Coder Python 1.73.1” on Ucloud.sdu.dk. The final step it to clone the GitHub repository on your own device.

### 5.2 Install packages
Run the command “bash setup.sh” from the command line to create a virtual environment and install the required packages:

            bash setup.sh

### 5.3 Run the scripts
To run emotion_clf.py, run the command “bash run.sh” from the command line. This activates the virtual environment, run the script, and deactivates the environment.

            bash run.sh

## 6. Discussing of the results

Most of the headlines in the total data set are *neutral*. The second most common emotion is *fear*, and the third is *anger*. *Sadness* is number four and it only occurs 53 times more than *disgust*. *Surprise* is number six and *joy* is number seven. So negative emotions dominate the total corpus of news articles. Only 363 of the 6335 headlines belongs to one of the two positive emotions joy and surprise. It has to be noted, that *surprise* is an ambiguous word that can both mean something positive and negative. 
	The distribution of emotions across the real headlines and the fake headlines shows the same tendency as the complete dataset: most headlines are *neutral*, and only few express *joy* or *surprise*. I expected the proportion of negative emotions among fake news headlines to be bigger than among real news headlines, because I was under the impression, that fake news is mostly used to spread negative rumors about other people and events, but according to this model, the negative emotions are not a special character trait for fake news. 

## 7. References
Fake or Real News, Kaggle: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news 

Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
