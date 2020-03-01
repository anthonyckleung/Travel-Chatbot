"""Travel Chat Bot on Slack

title:  chat_endpoint.py
author: Anthony Leung
usage:  python3 chat_endpoint.py
"""

import os
import random 
import pandas as pd
import json
import dateparser
import spacy
import response
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher

from fastai import *
from fastai.text import *
from fastai.callbacks import *

from slackeventsapi import SlackEventAdapter
from slack import WebClient

# Change directory to parent folder
os.chdir('../')

# Our app's Slack Event Adapter for receiving actions via the Events API
slack_signing_secret = os.environ['SLACK_SIGNING_SECRET']
slack_events_adapter = SlackEventAdapter(slack_signing_secret, '/slack/events')

# Create a SlackClient for your bot to use for Web API requests
slack_bot_token = os.environ['SLACK_BOT_TOKEN']
slack_client = WebClient(slack_bot_token)

slack_bot_id = os.environ['SLACK_BOT_USER_ID']

####==== SKYSCANNER ====####
sky_url = os.environ['SKYSCAN_URL']
rapid_host = os.environ['RAPID_HOST']
rapid_key = os.environ['RAPID_KEY']

####===== spaCy ====####
# Load spaCy object
nlp = spacy.load('en_core_web_sm')

# Create Matcher object for phrase matching
matcher = Matcher(nlp.vocab)

# Starting location
pattern_start = [{'LOWER': 'from', },
            {"ENT_TYPE": "GPE", "OP": "+"},
            ]

# Ending/Destination location
pattern_end = [{'LOWER': 'to'},
            {"ENT_TYPE": "GPE", "OP": "+"}]

# Add patterns to matcher object
matcher.add("START_LOC", None, pattern_start)
matcher.add("END_LOC", None, pattern_end)



####===== Process chatbot responses =====#####
chat_df = pd.read_csv('travel_chat.csv')
chat_df['response'] =  chat_df['response'].apply(lambda x: x.strip('[]')
                                          .replace("'","").split(', '))
response_dict = dict(zip(chat_df['label'], chat_df['response'].tolist()))

####===== Load and process Airport codes ====#####
fields = ['Name', 'City', 'IATA']
airport_df = pd.read_csv('airports.csv', usecols=fields)

air_int_df = airport_df[airport_df['Name'].apply(lambda x: 'International' in x)]

####===== Load speech intent classifier =====#####
data_bunch = 'data_clas_export_travel_chats.pkl'
trained_model ='travel-chat-clas-model'
encoder = 'ft_enc'

path = Path('')
data_clas = load_data(path, data_bunch, bs=8)
clas_learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
clas_learn.load(trained_model)
clas_learn.load_encoder(encoder)


####==== HELPER METHODS ====####
def ner_doc(doc):
    '''
    takes an spacy doc object and makes a ner 
    and returns a dataframe of entities.
    '''
    col_names = ['text',  'label']
    sent_df = pd.DataFrame(columns=col_names)
    for ent in doc.ents:
        temp = pd.DataFrame([[ent.text, ent.label_]], columns=col_names)
        sent_df = pd.concat([sent_df, temp], ignore_index=True)
    return sent_df 


def loc_matcher(doc):
    '''
    Takes a spacy doc object and find  
    matching patterns on origin and destination
    locations. Returns results as a dataframe.
    '''
    #match_id, start and stop indexes of the matched words
    matches = matcher(doc)
    col_names = ['pattern', 'text', 'location']
    match_result = pd.DataFrame(columns=col_names)

    #Find all matched results and extract out the results
    for match_id, start, end in matches:
        # Get the string representation 
        string_id = nlp.vocab.strings[match_id]  
        span = doc[start:end]  # The matched span
        loc = doc[start+1:end].text
        temp = pd.DataFrame([[string_id, span.text, loc]], columns=col_names)
        match_result = pd.concat([match_result, temp], ignore_index=True)
        
    return match_result

def travel_api_get(ner_df, match_df):
    '''
    Makes a GET request on a travel API to get flight information.
    '''
    # Extract all relevant entities
    origin_list = match_df[match_df['pattern']=='START_LOC'].loc[:,'location'].tolist()
    dest_list = match_df[match_df['pattern']=='END_LOC'].loc[:,'location'].tolist()
    
    origin_loc = origin_list[-1]
    dest_loc = dest_list[-1]
    
    # Map location to airport code
    origin_code = air_int_df[air_int_df['City'] == origin_loc]['IATA'].iloc[0]
    dest_code = air_int_df[air_int_df['City'] == dest_loc]['IATA'].iloc[0]

    # Parse any date entities to the format 'YYYY-MM-DD'
    depart_date = None
    return_date = None
    date_df = ner_df[ner_df['label'] == 'DATE']
    n_date = len(date_df)
    if (n_date == 2):
      date_df = date_df.sort_values(by=['text'])
      depart_date = date_df['text'].iloc[0]
      return_date = date_df['text'].iloc[1]
     
      depart_date = dateparser.parse(depart_date).strftime('%Y-%m-%d')
      return_date = dateparser.parse(return_date).strftime('%Y-%m-%d')
    else:
      depart_date = date_df['text'].iloc[0]
      depart_date = dateparser.parse(depart_date).strftime('%Y-%m-%d')
    
    # Define url properties
    URL = sky_url 
    country = 'CA'
    currency = 'CAD'
    locale = 'en-US'
    originPlace = origin_code
    destinationPlace = dest_code
    outboundPartialDate = depart_date

    URL_complete = f'{URL}/{country}/{currency}/{locale}/{originPlace}/{destinationPlace}/{outboundPartialDate}'
    
    headers = {
    'x-rapidapi-host': rapid_host,
    'x-rapidapi-key': rapid_key 
    }
    # JSON object output from the GET requet has 'true' and 'false' 
    # entries in improper format. Need to set them accordingly.
    true = True
    false = False

    # If there is a return date, then add inbound date into URL
    if return_date != None:
      inboundPartialDate = return_date
      URL_complete = URL_complete + f'/{inboundPartialDate}'
    
    # Pull flight info via GET request
    flight_resp = requests.request('GET', URL_complete, headers=headers)

    return flight_resp


def flight_response(doc):
    '''
    Constructs a response for a flight request from the user.
    '''
    # Identify relevant entities
    ner_df = ner_doc(doc)

    # Get the number of dates and locations
    n_loc = len(ner_df[ner_df['label'] == 'GPE'])
    n_date = len(ner_df[ner_df['label'] == 'DATE'])

    # Return if there are not enough flight information.
    if (n_loc < 2) or (n_date ==0):
        resp = "Sorry I don't understand. Please restate your flight request \
with complete dates and locations."
        return resp

    # Continue routine via GET request
    match_df = loc_matcher(doc)
    flight_resp = travel_api_get(ner_df, match_df)
    resp_json = json.loads(flight_resp.text)

    # Extract all price quotes
    quotes = resp_json['Quotes']
    n_quotes = len(quotes)
    if (n_quotes == 0):
        resp = 'No flights are available with the details you provided.'
        return resp
    
    # Concat prices into a single string.
    price_str = ""
    for quote in quotes:
        price = quote['MinPrice']
        depart_time = quote['OutboundLeg']['DepartureDate']
        price_str += f'${price} ({depart_time}); '

    #Extract all relevant flight entities
    depart_date = ner_df.loc[ner_df['label'] == 'DATE', 'text'].iloc[0]
    
    start_list = match_df[match_df['pattern']=='START_LOC'].loc[:,'location'].tolist()
    dest_list = match_df[match_df['pattern']=='END_LOC'].loc[:,'location'].tolist()
    
    origin_loc = start_list[-1]
    dest_loc = dest_list[-1]
    
    resp = f'Flight tickets from {origin_loc} to {dest_loc} leaving on {depart_date}: '+price_str
    
    return resp 
       

@slack_events_adapter.on('message')
def handle_message(event_data):
    '''
    Main method to handle Slack messages.
    '''
    message = event_data['event']
     
    #If the incoming message is not from the bot, then respond. 
    if message['user'] != slack_bot_id:
       msg = message.get('text')
       pred = clas_learn.predict(msg)
       msg_intent = str(pred[0])
       channel = message['channel']
       doc = nlp(msg)
       resp = random.choice(response_dict[msg_intent])
     
       if (msg_intent=='SearchFlight'):
            resp = flight_response(doc)

       slack_client.chat_postMessage(channel=channel,
                                  text=resp)
                               

if __name__=="__main__":

      slack_events_adapter.start(port=3000)
