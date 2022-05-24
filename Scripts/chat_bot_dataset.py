# Import relevant packages:
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk import PorterStemmer, tokenize
import json
import boto3

class ChatbotDataset(Dataset):
    def __init__(self):
        self.clean_data()


    def __getitem__(self, idx): 
        pass

    def __len__(self):
        pass 
    
    def get_raw_data(self, data_path):
        with open(data_path, 'r') as f:
            message_json = json.loads(f.read())

        return message_json

    def clean_data(self):
        # For each datapoint we need to: drop "xoxox", tokenize, lower, stem, turn into a vector using bag_of_words.   
        raw_data = self.get_raw_data('../Data/message_data.json')

        #set_labels = set([message_group['tag'] for message_group in raw_data['messages']])
        #self.label_mapping = {j: i for i,j in enumerate(set_labels)}
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


    def bag_words(self):
        self.clean_data()
        pass

