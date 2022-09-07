# Coronavirus-Tweet-Sentiment-Analysis
Predictive model for Covid-19 tweet sentiment analysis based on the tweet dataset from January 4, 2020 to December 4, 2020

## Introduction

Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the
writer’s attitude towards a particular topic is Positive, Negative, or Neutral.

Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.

## PROBLEM STATEMENT

We are provided with a coronavirus tweets csv file which contains more than 40000 tweets from people around the world on covid 19 and our aim is to analyze these
tweets made on Covid-19 from around the world and predict the sentiment of each of the tweet by classifying them into three categories positive, negative and neutral.

## OBJECTIVE

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

● Analysed hashtags and word count.

● The majority of the tweets in our dataset had a positive attitude.

● Most of the tweets in our record are from March, whereas for all other months, the number of tweets is more or less constant.

● On average, positive sentiment tweets have a higher word count than negative sentiment tweets. Neutral Sentiment tweets have a much lesser word count than positive and negative sentiment tweets. 

● In all the tweets, irrespective of the sentiment the most frequently used words apart from the name of the disease are supermarket, grocery store, toilet paper, online shopping, food and price which signals how concerning it was for the people even to get basic day to day items during the pandemic.


## Pre-processing of data

● Removed usernames and links using regex and then performed a tokenization process.

● Fixed the contractions using the contraction library

● Removed stopwords and punctuation and applied lower casing.

● Lemmatization was used to transform the words into their root form.

● Converted the target variable into numeric by assigning 1 for positive, 0 for neutral, and -1 for negative tweets.

● Finally, vectorized the tokens using TFID Vectorizer to make the text data machine readable.

## Models Used

● Naive Bayes 

● Logistic Regression

● CatBoost 

● Stochastic Gradient Descent 

● Linear Support Vector Machine

## Conclusion

● Among the 5 classification algorithms we performed SVM, SGD, Logistic Regression, and CatBoost perform well on the test data, SVM being the best model.

● Linear SVM with 85% accuracy after cross-validation and hyperparameter tuning performed well for multinomial classification.

● We hope this project will be helpful for the Government and NGOs to take adequate measures in policy making and rehabilitation respectively.

● Various profit organizations can make profit through the production and distribution of essential items during the pandemic by analyzing various sentiments.
