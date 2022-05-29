# Import Packages:
import torch
import torch.nn as nn
import os 
from chat_bot_dataset import ChatbotDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import NeuralNet
from torch.utils.tensorboard import SummaryWriter

#Create Writer:
writer = SummaryWriter()

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

# Create Dataloader to pass data in batches:
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

#Initialise Model:
model = NeuralNet(input_size, hidden_layer_1, hidden_layer_2, num_classes)

#Initialise Optimizer and loss(recall - CrossEntropy applies Crossmax for us):
optimizer = Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

#Training Loop:
for epoch in range(num_epochs):
    for batch_no, (data, labels) in enumerate(train_loader):
        batch_no += 1
        predictions = model(data)
        error = loss(predictions, labels)
        print(error)
        writer.add_scalar('Loss/Train', error, epoch)
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

writer.flush()