#Import Packages:
import torch
import json
from Scripts.model import NeuralNet
from Scripts.training import absolute_path
from Scripts.text_cleaning import clean_text, bag_of_words


# Load in the trained model:
chatbot_model = torch.load(absolute_path)

#Store the contents of the model dictionary:
num_features = chatbot_model['input_size']
hidden_layer_1 = chatbot_model['hidden_size_1']
hidden_layer_2 = chatbot_model['hidden_size_2']
num_classes = chatbot_model['output_size']
bag = chatbot_model['bag']
label_mapping = chatbot_model['label_mapping']
trained_params = chatbot_model['net']
raw_data = chatbot_model['raw_data']

#Load in an untrained model:
model = NeuralNet(num_features, hidden_layer_1, hidden_layer_2, num_classes)

#Change randomised model parameters to trained params:
model.load_state_dict(trained_params)

# Set model to evaluation mode:
model.eval()