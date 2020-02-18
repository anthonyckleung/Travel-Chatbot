from fastai import *
from fastai.text import *
from fastai.callbacks import *
import random 
import pandas as pd

from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher

from slackeventsapi import SlackEventAdapter
from slack import WebClient
import os

# Our app's Slack Event Adapter for receiving actions via the Events API
slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
slack_events_adapter = SlackEventAdapter(slack_signing_secret, "/slack/events")

# Create a SlackClient for your bot to use for Web API requests
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
slack_client = WebClient(slack_bot_token)

####===== spaCy ====####
# Load spaCy object
nlp = spacy.load('en_core_web_sm')
# Create Matcher object for phrase matching
matcher = Matcher(nlp.vocab)

####===== Define Patterns for SpaCy Phrase Matching ====####
# Starting pattern
matcher.add("START_LOC", None, pattern_start)

# Destination pattern
matcher.add("END_LOC", None, pattern_end)



####===== Process chat responses =====#####
chat_df = pd.read_csv('travel_chat.csv')
chat_df['response'] =  chat_df['response'].apply(lambda x: x.strip('[]')
                                          .replace("'","").split(', '))
response_dict = dict(zip(chat_df['label'], chat_df['response'].tolist()))

#print(response_dict)

####===== Load intent classifier =====#####
data_bunch = 'data_clas_export_travel_chats.pkl'
trained_model ='travel-chat-clas-model'
encoder = 'ft_enc'

path = Path('')
data_clas = load_data(path, data_bunch, bs=8)
clas_learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
clas_learn.load(trained_model)
clas_learn.load_encoder(encoder)


#temp = clas_learn.predict('Hi there')

#class_predict = str(temp[0])
#print(class_predict)
#print(random.choice(response_dict[class_predict]))

def ner_doc(doc):
  col_names = ['text',  'label']
  sent_df = pd.DataFrame(columns=col_names)
  for ent in doc.ents:
    temp = pd.DataFrame([[ent.text, ent.label_]], columns=col_names)
    sent_df = pd.concat([sent_df, temp], ignore_index=True)
  return sent_df 





# Example responder to greetings
@slack_events_adapter.on("message")
def handle_message(event_data):
    message = event_data["event"]
    #print(event_data) 
    #If the incoming message is not from the bot, then respond. 
    if message['user'] != 'UU1PNRKUG':
       msg = message.get('text')
       pred = clas_learn.predict(msg)
       msg_intent = str(pred[0])
       channel = message['channel']
       resp = random.choice(response_dict[msg_intent])
       slack_client.chat_postMessage(channel=channel,
                                  text=resp)
                               

if __name__=="__main__":
    slack_events_adapter.start(port=3000)
