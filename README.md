# 3 Ways You Can Improve Your BERT Model

1. Set up Optimizer
2. Adjust Scheduler 
3. Fine-tuning no. of epochs

## Table of contents
* [About the Project](#about-the-project)
* [Description](#description)
* [Getting Started](#getting-started)
  + [Prerequisites](#prerequisites)
  + [Installation](#installation)
* [Steps](#steps)
* [Acknowledgements](#acknowledgements)

## About the Project
This project is about finetuning BERT to do text classification, highlighting 3 finetuning methods.

## Description
- Classification of tweets according to emotions towards arts and cultural experiences in museums

- Dataset:
  - 3085 tweets
  - 5 emotions: anger, disgust, happiness, surprise, sadness

![github](https://user-images.githubusercontent.com/55085035/111116043-67f05f80-85a0-11eb-8dfe-2573e1ff9985.png)

## Getting Started

### Prerequisites
- Basic understanding of Deep Learning and NLP models (BERT)

### Installation

- Clone the repo:

```
$ git clone https://github.com/weiling97/textclassification_bert.git
```

## Steps

### Step 1: Exploratory Data Analysis & Preprocessing
- Remove categories with multiple emotions, & without emotion
- Assign class labels to the emotions. E.g. happy: 0, surprise: 5 

### Step 2: Training/Validation Split
- Import sklearn module to split data into training and validation data
![Picture 1](https://user-images.githubusercontent.com/55085035/111117262-42645580-85a2-11eb-8152-583d7a672ab8.png)


### Step 3: Loading Tokenizer and Encoding our Data
- Tokenize text, convert text into lower case
- Encode data, i.e. converting tweets into numbers. E.g. dog == 102, cat == 256
- [Attention mask](https://github.com/huggingface/transformers/issues/205#:~:text=It%27s%20a%20mask%20to,batch%20has%20varying%20length%20sentences.) is applied here. Used if length < max length of tweet (used when there's varying length to sequences)
- Thus, creating training and validation datasets

### Step 4: Setting up BERT Pretrained Model
- Download base pretrained bert model (import BertForSequenceClassification)
- “bert-base-uncased” is used

### Step 5: Creating Data Loaders
- Load the data while sampling the elements randomly

### Step 6: Setting Up Optimizer and Scheduler
- Optimizer and scheduler are parts of what makes BERT works
- **Optimizer** uses Adam algorithm with weight to define how learning rate changes through time
- **Scheduler** means what controls the learning rate and define the number of training steps here
- E.g. if lr = 2e-5, during training the lr will be linearly increased from approximately 0 to 2e-5 within the first 10,000 steps

### Step 7: Defining our Performance Metrics
- Use f1 score and accuracy per class methods
- Use f1 score cause there's a class imbalance since simply using accuracy might give a skewed result

### Step 8: Creating our Training Loop
- Tune BERT for 10 epochs: learning algorithm will work through the entire training set 10 times
- During training, we evaluate our model parameters against the validation data
- We save the model each time the validation loss decreases so that we end up with the model with the lowest validation loss -> best model
- Batch size is the number of samples to work through before updating the internal model parameters
- Loss is the result of bad prediction


## Acknowledgements

