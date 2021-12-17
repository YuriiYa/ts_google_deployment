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

try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=True
  )
except ImportError:
  pass
  

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
    
#copy file from bucket
model_dest_path = '/tmp/pytorch_model.bin';
download_blob('tinder-standup-model-bucket1','pytorch_model.bin',model_dest_path)

rootPath ="./"

#these are buried in the src directory for transformers. If you want to look at 
#the scripts, run the block where they are used, and click on popup box. 
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline

#reloading model and tokenizer from local storage

print(model_dest_path)
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model=GPT2LMHeadModel.from_pretrained(model_dest_path)

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