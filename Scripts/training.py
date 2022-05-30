# Import Packages:
import torch
import torch.nn as nn
import copy
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
nn_model = NeuralNet(input_size, hidden_layer_1, hidden_layer_2, num_classes)

#Initialise Optimizer and loss(recall - CrossEntropy applies Softmax for us):
optimizer = Adam(nn_model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

#Training Loop:
def train(model, optim, loss_func, number_of_epochs=100):

    #Want to save the model with the highest accuracy on training set:
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(number_of_epochs):

        epoch_accuracy = 0
        model_state = copy.deepcopy(model.state_dict())

        for batch_no, (data, labels) in enumerate(train_loader):
            batch_no += 1
            predictions = model(data)
            error = loss_func(predictions, labels)
            _, class_predictions = torch.max(predictions, axis=1)
            epoch_accuracy += torch.sum(class_predictions == labels)
            writer.add_scalar('Loss/Train', error, epoch)
            error.backward()
            optim.step()
            optim.zero_grad()
        
        if epoch_accuracy >= best_acc: 
            best_model = model_state

    # Load the model with the highest testing accuracy
    model.load_state_dict(best_model)

    writer.flush()

    return model

train(nn_model, optimizer, loss, number_of_epochs=num_epochs)