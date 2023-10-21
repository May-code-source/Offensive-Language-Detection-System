# Offensive-Language-Detection-System

## Table of content

- [Overview](#overview)
- [Highlight](#highlight)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Conclusion](#conclusion)


## Overview

Cyberbullying has become a severe issue, facilitated by the growth of social media. It can have devastating psychological impacts on victims. This project involved building a multi-modal machine learning system to detect offensive language in social media posts and chat messages, which are commonly used in the case of cyberbullying. The system utilizes natural language processing and deep learning for real-time analysis.

## Highlight
- Preprocessed data including emoji handling😆, their conversion into textual form, cleaning, EDA, and data augmentation to address class imbalance
- Morphological Analysis
- Sentiment Analysis
- Stratified Cross Validation
- Optuna Hyperparemeter Tunning  
- Compared SVM, LSTM, and BERT models
- BERT achieved the best accuracy of 94% on the test set
- Integrated model into chat interface with real-time analytics

## Dataset

- Collected a diverse dataset of 10,000 tweets that encompass emojis 😃 and across various topics. Data was sourced from Twitter and Kaggle 

## Evaluation
- Accuracy, precision, recall, F1 score, ROC AUC
- Confusion matrix, precision-recall curve
- Learning curves

![Picture3](https://github.com/May-code-source/Offensive-Language-Detection-System/assets/115402970/452f31cc-632d-4a75-ad36-4a228f91c465)


[View Jupyter Note Here](https://nbviewer.org/github/May-code-source/Offensive-Language-Detection-System/blob/main/Detection_Script.ipynb)

## Deployment
- Flask backend framework
- SocketIO enabled real-time analysis
- Proactive intervention enabled by real-time analysis to foster positive interactions through increased awareness and deterrence of negative content
- Chat interface with sentiment chart, timing data, and warnings
- Chat interface showcases real-world applicability.

## Conclusion
The system successfully detects cyberbullying instances in text and emojis😃 using deep learning.
