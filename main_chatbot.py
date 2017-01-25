import json
import requests
import time
import urllib
from gensim import utils, matutils, models
from sklearn.externals import joblib
from secret import *



URL = "https://api.telegram.org/bot{}/".format(TOKEN)

def get_url(url):
    ''' Gets url that returns content '''
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    '''Gets json from url '''
    content = get_url(url)
    js = json.loads(content)
    return js

def get_updates(offset=None):
    ''' Gets the updates from the Telegram server '''
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js


def get_last_update_id(updates):
    ''' Gets the ID of the last update '''

    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)

def get_last_chat_id_and_text(updates):
    ''' Gets the last chat ID and the text '''

    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)

def send_message(text, chat_id):
    ''' Sends message to client '''
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)


def echo_all(updates):
    ''' Tests echoing messages received '''
    for update in updates["result"]:
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        send_message(text, chat)

def basic_responses(updates):
    ''' Tests basic rule-set for sending messages '''
    for update in updates["result"]:
        chat = update["message"]["chat"]["id"]
        text = update["message"]["text"]
        if text.lower() == 'hello':
            response = 'Hello there'
        elif text.lower() == 'how are you?':
            response = "Very well thankyou"
        else:
            continue
        send_message(response, chat)

def doc_response(updates):
    ''' Generates responses to messages recived '''

    # For each message
    for update in updates["result"]:
        # Extract chat_id and text
        chat = update["message"]["chat"]["id"]
        text = update["message"]["text"]
        # Turn input text into a vector and reshape
        vector = doc_model.infer_vector(utils.simple_preprocess(text), steps = 40)
        vector = vector.reshape(1, -1)
        # Predict the appropriate response label
        prediction = logistic_model.predict(vector)
        # Use label to exstract the text from the array of all possible responses
        response = possible_responses[prediction[0]]
        # Send message to client
        send_message(response, chat)





def main():
    ''' Runs the chatbot functionality in a loop '''
    last_update_id = None
    # Loop indefinitel
    while True:
        # Get the latest messages from client
        updates = get_updates(last_update_id)
        # If they exist
        if len(updates["result"]) > 0:
            # Set last_update_id
            last_update_id = get_last_update_id(updates) + 1
            # Generate response to message
            doc_response(updates)
        # Sleep program for half a second
        time.sleep(0.5)

if __name__ == '__main__':
    # Loads saved models from disk
    doc_model = models.Doc2Vec.load(doc2vec_location)
    model_details = joblib.load('model.pkl')
    logistic_model = model_details['model']
    possible_responses = model_details['response_vector']
    main()