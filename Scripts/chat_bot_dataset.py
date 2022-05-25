# Import relevant packages:
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import boto3

class ChatbotDataset(Dataset):
    def __init__(self):
        # For each datapoint we need to: drop "xoxox", tokenize, lower, stem, turn into a vector using bag_of_words.  
        self.extract_data()


    def __getitem__(self, idx): 
        raw_message = self.messages_list[idx]
        tokenized_message = word_tokenize(raw_message)

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
        label_index = 0

        # list to store each message in dataset:
        self.messages_list = []

        #list to store message_labels:
        self.labels = []
        
        for message_group in raw_data['messages']:
            if message_group['tag'] not in self.label_mapping:
                self.label_mapping[message_group['tag']] = label_index
                label_index += 1
            
            self.messages_list.extend(message_group['keywords'])
            self.labels.extend([message_group['tag']] * len(message_group['keywords']))
            
        self.numeric_labels = [self.label_mapping[label] for label in self.labels]


    def clean_data(self):
        # lower case all messages:
        self.messages_list = [message.lower() for message in self.messages_list]
        

    def bag_words(self, tokenized_text):
        pass
