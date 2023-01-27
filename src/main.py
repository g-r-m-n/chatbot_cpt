# -*- coding: utf-8 -*-

# create requirements:
# !pipreqs pth_to_src

# %% load libraries
import numpy as np
import pandas as pd
import sys, time, os
from datetime import date
from datetime import datetime
# determine the path to the source folder 
pth_to_src = 'C:/DEV/chatbot_cpt/src/'

# date of today:
today = date.today().strftime('%Y%m%d')

# input folder:
input_folder  = pth_to_src + 'input/'

# output folder:
output_folder = pth_to_src+ 'output/'+today+'/' # with date of today. This way a daily history of results is automatically stored.
output_folder_plots  = output_folder+'plots/'
output_folder_model  = output_folder+'model/'
# create output_folder if not existant:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
os.makedirs(output_folder_model,exist_ok=True)
# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(888) # set random seed for reproduceability


# %% Connect to api:
# the api key
from private.keys import API_SECRETE_KEY
import os
import openai
openai.api_key_path = pth_to_src+'private/apikey.txt'
openai.organization = ''
openai.api_key = os.getenv(openai.api_key_path)
openai.Model.list()


# %%


# list engines
engines = openai.Engine.list()

# sorted list of engine names:
engine_names= [i['id'] for i in engines.data ] ; engine_names.sort()

# print the engine names
    
print(engine_names)

# create a completion
completion = openai.Completion.create(engine="text-davinci-003", prompt="Hello world")

# print the completion
print(completion.choices[0].text)



# %% Embeddings
# =============================================================================
# In the OpenAI Python library, an embedding represents a text string as a fixed-length vector of floating point numbers. Embeddings are designed to measure the similarity or relevance between text strings.
# 
# To get an embedding for a text string, you can use the embeddings method as follows in Python:
# =============================================================================
    
# choose text to embed
text_string = "sample text"

# choose an embedding
model_id = "text-similarity-davinci-001"

# compute the embedding of the text
embedding = openai.Embedding.create(input=text_string, engine=model_id)['data'][0]['embedding']


# %% Moderation
moderation_resp = openai.Moderation.create(input="Here is some perfectly innocuous text that follows all OpenAI content policies.")

# https://beta.openai.com/docs/guides/moderation


# %% Image creation (DALL·E)
image_resp = openai.Image.create(prompt="two dogs playing chess, oil painting", n=4, size="512x512")