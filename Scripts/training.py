# Import Packages:
import torch
import torch.nn as nn
import os 
from chat_bot_dataset import ChatbotDataset
from torch.utils.data import DataLoader

#Initialise Training Dataset:
train_data = ChatbotDataset()

#Model Hyperparams: 
input_size = train_data.bag_size
hidden_layer_1 = 10
hidden_layer_2 = 6
num_classes = train_data.num_classses
batch_size = 6
learning_rate = 0.001
num_epochs = 1000