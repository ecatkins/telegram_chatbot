# Telegram Chatbot

This program creates a "smart" [Telegram] (https://telegram.org/) Chatbot, using the Telegram API, a Doc2Vec language model, and a basic Logistic Regression classification algorithm.

The creation of the parts of the program that interact with the Telegram API utilised information found in [this] (https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay) tutorial.

### Requirements
* numpy, pandas, requests, sklearn, gensim & a pre-trained gensim Doc2Vec model
* set-up a secret.py file according to the _secret.py template

### Description of project basics
The chatbot created is a retrieval based model, rathern than a generative model. This means that it selects from a range of pre-defined answers rather than trying to generate novel responses to the input of the user.

A pre-trained Doc2Vec model was used to generate feature vectors from text. A basic taxonomy ('harcode_examples.csv') of example user inputs and ideal responses were used to train a logistic regression model, with the X variables the features generated by the Doc2Vec model for a given input, and the Y variable a numerical categorial label assigned to the ideal output. The model is 'taught' to select the best response for a given user input.

### Results

The GIF below shows a simple interaction with the developed chatbot. The model has not been trained directly on any of these exact inputs, but it is able to select a suitable response in reply. For example the model was not trained on the phrase "G'day amte" (sic), but the power of the Doc2Vec model is that is able to determine it is similar to the phrases "Hello" and "Howdy", which it has seen before, and knows to generate "Hello yourself" in response.

![](http://g.recordit.co/fNasb139vs.gif)

### Potential Extensions
1. More training examples (currently only 17)
2. Store entire context of conversation, or utilise model with some hidden state, that tracks this kind of information
3. Generative model 
