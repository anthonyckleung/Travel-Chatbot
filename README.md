# Travel Chatbot on Slack

The purpose of this project is to build a simple chatbot where a user can request flight prices through conversation.

The project was originally assigned at 1000ml - an institution for practical AI based in Toronto. Presentation of results can be found in this repo as well. Note that work is added onto this repo in a ongoing basis and so the slides represents progress at an earlier time. 

## Prerequsite
Notebook and script are written in Python 3. The libraries used for this project are listed under the `requirements.txt` file. It is recommended to first set up a virtual environment and then install the libraries with:

```bash
pip3 install -r requirements.txt
```

## Description
A speech intent classifier is trained by using NLP transfer learning - the Universal Language Model Fine-tuning (ULMFiT) method. I can upon this approach when researching transfer learning in NLP applications since it already existed for image classification. In this work, ULMFiT was implemented by using fastai: https://docs.fast.ai/text.html. An impressive accuracy of 93% was achieved on the speech intent classifier considering my travel chat corpus has about 250 examples. The work is documented in the `chat_model.ipynb` notebook in this repo. It should be noted that the notebook was originally developed on Google Colab.

An endpoint of sorts is created which contains a travel API and the SlackEventsAPI. This is referring to the `chat_endpoint.py` script in this repo. The `requests` library on Python is used for http methods to make requests with the travel API. Furthermore, one would need to create an app on Slack in order to generate the required tokens that allows the script to interact with Slack. One can follow the Slack [tutorial](https://github.com/slackapi/python-slack-events-api/tree/master/example) to set this up. 

## Demo
![](web/chat_demo.gif)
