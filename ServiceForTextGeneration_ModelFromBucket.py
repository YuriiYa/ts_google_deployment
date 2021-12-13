# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 06:57:26 2021

@author: yuyan
"""
#installing dependencies 
#!pip install -q transformers
# conda install -c conda-forge transformers
# pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
#|?

#importing
import os 
import shutil
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import transformers
import torch
from google.cloud import storage

#copy file from bucket
storage_client = storage.Client()
bucket = storage_client.get_bucket('tinder-standup-model-bucket')
model_fileName= 'pytorch_model.bin'
blob = bucket.blob(model_fileName)
model_path= os.path.join('/tmp',model_fileName);
blob.download_to_filename(model_path)


#these are buried in the src directory for transformers. If you want to look at 
#the scripts, run the block where they are used, and click on popup box. 
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline
rootPath ="./"
workingPath=rootPath

#reloading model and tokenizer from local storage

print(model_path)
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
#tokenizer=GPT2Tokenizer.from_pretrained(model_path)
model=GPT2LMHeadModel.from_pretrained(model_path)

from flask import Flask,request
import pickle
import numpy as np

app = Flask(__name__)
model.config.max_length=1000
model.config.temperature=1.3

# http://localhost:5000/api_predict
# " Cinderella start programming to  "
# {"input":" Home is not only cosiness but ","max_length":1000,"temperature":1.3}

@app.route('/api_generate', methods=["GET","POST"])
def api_predict():
    if request.method =="GET":
        return "Please send Post Request"
    elif request.method=="POST":
        data = request.get_json()
        
        model.config.max_length=data['max_length']
        model.config.temperature=data['temperature']
        context=data['input']
        #model.config.max_length=1000
        #model.config.temperature=1.3
        #context=" Home is not only cosiness but "
        tokens=tokenizer.encode(context)
        
        #pipelines in the transformers library combine model and tokenizer in one. 
        pipe=pipeline("text-generation", model, tokenizer=tokenizer)
        output=pipe(context)
        return f"{output[0]['generated_text']}"

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)