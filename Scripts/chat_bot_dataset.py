# Import relevant packages:
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import boto3

class ChatbotDataset(Dataset):
    def __init__(self):
        # For each datapoint we need to: drop "xoxox", tokenize, lower, stem, turn into a vector using bag_of_words.  
        self.ps = PorterStemmer()
        self.extract_data()
        self.clean_data()



    def __getitem__(self, idx): 
        pass

    def __len__(self):
        pass 
    
    def get_raw_data(self, data_path):
        with open(data_path, 'r') as f:
            message_json = json.loads(f.read())

        return message_json

    def extract_data(self): 
        raw_data = self.get_raw_data('../Data/message_data.json')

        # Dictionary to store numerical encoding of labels.
        self.label_mapping = {}
        label_index = 1

        # list to store each message in dataset:
        self.messages_list = []

        #list to store message_labels:
        self.labels = []
        
        for message_group in raw_data['messages']:
            # create a numerical encoding for message tags.
            if message_group['tag'] not in self.label_mapping:
                self.label_mapping[message_group['tag']] = label_index
                label_index += 1
            
            self.messages_list.extend(message_group['keywords'])
            self.labels.extend([message_group['tag']] * len(message_group['keywords']))
        
        # Get numerical encoding of labels
        self.numeric_labels = [self.label_mapping[label] for label in self.labels]


    def clean_text(self, text):
        lowered_text = text.lower()
        list_text = lowered_text.split(' ')
        clean_text = [self.ps(word) for word in list_text]
        return clean_text 
        

    def clean_data(self):
        # Clean up all messages and extracting "bag" for bag of words:
        self.bag = set()

        # list to hold cleaned messages:
        self.cleaned_messages_list
        for message in self.messages_list:
            cleaned_message = self.clean_text(message)
            self.cleaned_messages = self.cleaned_messages_list.append(cleaned_message)
            self.bag.update(set(cleaned_message))
        self.bag = list(self.bag)

    def bag_words(self, tokenized_text):
        pass
