# Coronavirus-Tweet-Sentiment-Analysis
Predictive model for Covid-19 tweet sentiment analysis based on the tweet dataset from January 4, 2020 to December 4, 2020

![image](https://user-images.githubusercontent.com/92729412/193468265-fc3cb3b6-f6d5-451f-b71c-eea0399d2f97.png)

## Introduction

Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the
writer’s attitude towards a particular topic is Positive, Negative, or Neutral.

Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.

## Problem Statement

We are provided with a coronavirus tweets csv file which contains more than 40000 tweets from people around the world on covid 19 and our aim is to analyze these
tweets made on Covid-19 from around the world and predict the sentiment of each of the tweet by classifying them into three categories positive, negative and neutral.

## Objectives

1. Analyze the tweets regarding COVID 19 and get insights regarding people’s sentiment.

2. To build a classification model to predict the sentiment of COVID-19 tweets which have been pulled from Twitter.

## Data Summary

The shape of the dataset is (41157, 6). The target variable is ‘Sentiment’.

● Username : The username of the person on twitter

● Screenname : The screenname of the person on twitter

● Location : The location from where the tweet was tweeted

● TweetAt : The date of the tweet

● OriginalTweet : The tweet itself unfiltered

● Sentiment : The sentiment of the tweet our target variable

## EDA

● Analysed hashtags,mentions and word count.

![image](https://user-images.githubusercontent.com/92729412/193468313-e4f9b9b0-0235-4077-9c79-22aef4a799c1.png)

The number of words present in tweets of neutral sentiment is far less when compared to tweets with a positive or negative sentiment . Most of the positive and negative sentiment tweets contain almost 40 words on average.

![image](https://user-images.githubusercontent.com/92729412/193468330-c28d9a6a-e5be-45b0-b01b-056cc38f75db.png)

Most of the tweets contain no mention and there is no particular relationship between number of mentions and regard the sentiment.

![image](https://user-images.githubusercontent.com/92729412/193468349-649f8ad0-8a21-4610-88a4-ff1063b80013.png)

Most tweets do not have a hashtag. Again, the number of hashtags has nothing to do with the sentiment.

![image](https://user-images.githubusercontent.com/92729412/193468408-1f1649ff-8269-4df1-b3ec-83f606004e14.png)

Most of the tweets in our record are from March, whereas for all other months, the number of tweets is more or less constant.

![image](https://user-images.githubusercontent.com/92729412/193468639-1b99df3d-d0fe-4ae8-b789-580f171914f6.png)

![image](https://user-images.githubusercontent.com/92729412/193468642-a82052e9-d021-4dbc-82dc-fde13e0487a6.png)

![image](https://user-images.githubusercontent.com/92729412/193468650-62959702-af98-4ffa-9669-ebc8de199825.png)

In all the tweets, irrespective of the sentiment the most frequently used words apart from the name of the disease are supermarket, grocery store, toilet paper, online shopping, food and price which signals how concerning it was for the people even to get basic day to day items during the pandemic.

## Pre-processing of Data

Before our tweets can be passed to machine learning algorithms, it needs some clean up or pre-processing so algorithms can focus on main/important words instead of words which add minimal to no value.

● Converted the target variable into numeric by assigning 1 for positive, 0 for neutral, and -1 for negative tweets.

● Removed usernames and links using regex and then performed a tokenization process. 

Some of our tweets contained HTML tags and usernames so we first started with removing them, HTML tags and mentions are typically one of these components which don’t add much value towards understanding and analyzing text so they should be removed. 

● Tokenization:

Tokenization is the process of splitting a text object into smaller units known as tokens. Examples of tokens can be words, characters, numbers, symbols, or n-grams.
The most common tokenization process is whitespace/ unigram tokenization. In this process the entire text is split into words by splitting them from whitespaces. The tokenization can be performed at the sentence level or at the word level or even at the character level.

● Fixed the contractions using the contraction library.

Contractions are shortened versions of words or syllables. They are created by removing specific, one or more letters from words. Often more than words are combined to create a contraction. In writing, an apostrophe is used to indicate the place of missing letters. Converting each contraction to its expanded, original form helps with text standardization.

● Removed stopwords and punctuation and applied lower casing.

Stopwords are often added to sentences to make them grammatically correct, for example, words such as a, is, an, the, and etc. These stopwords carry minimal to no importance and are available in plenty of open texts, articles, comments etc. We removed stopwords so that our  machine learning algorithms can better focus on words which define the meaning/idea of the text. 

An important NLP preprocessing step is punctuation marks removal, this marks - used to divide text into sentences, paragraphs and phrases. Removing punctuation marks are usually applied to reduce the feature size and improve the classification accuracy.

● Lemmatization was used to transform the words into their root form.

Lemmatization is a very popular and common text pre-processing technique. It will help us to group different inflicted forms of words into the root form called lemma which carries the same meaning. It helped us to diminish the number of tokens required to transfer the same information and so boosted up our model training.

● Finally, vectorized the tokens using TFID Vectorizer to make the text data machine readable.

To convert the text data into numerical data, we need some smart ways which are known as vectorization, or in the NLP world, it is known as Word embeddings. Therefore, Vectorization or word embedding is the process of converting text data to numerical vectors.

* TFIDF Vectorizer:
In information retrieval, tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.


## Models Used

● Naive Bayes 

● Logistic Regression

● CatBoost 

● Stochastic Gradient Descent 

● Linear Support Vector Machine

## Evaluation 

![image](https://user-images.githubusercontent.com/92729412/193468706-2f866670-bfe4-4ab3-8408-f5f5798850fd.png)

* Naive Bayes fails to perform well in the classification as its scores are low. * All the other four models, SVM, CatBoost, Logistic Regression and SGD Classifier perform well with the mertics score in the range of 0.75-0.82.
* Among these four models, SVM has the highest accuracy score. Its precision score, recall score and f1 score are also pretty good. So we conclude that SVM is the best performing model.

Let's do cross validation and hyperparameter tuning for the best model, SVM

![image](https://user-images.githubusercontent.com/92729412/193468745-4a9a765f-f1b2-4787-b6b7-088175204d15.png)

![image](https://user-images.githubusercontent.com/92729412/193468777-34509353-46e5-4a96-a8e4-c36e05f5ae48.png)

## Conclusion

● Among the 5 classification algorithms we performed SVM, SGD, Logistic Regression, and CatBoost perform well on the test data, SVM being the best model.

● Linear SVM with 84% F1-score after cross-validation and hyperparameter tuning performed well for multinomial classification.

● This project will be helpful for the Government and NGOs to take adequate measures in policy making and rehabilitation respectively.

● Various profit organizations can make profit through the production and distribution of essential items during the pandemic by analyzing various sentiments.

### This project will help the government authorities to understand the needs of the people during the pandemic and take timely action against fake news that creates unnecessary panic. 
