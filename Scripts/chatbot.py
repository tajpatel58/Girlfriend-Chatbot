#Import Packages:
import torch
import json
from Scripts.model import NeuralNet
from Scripts.training import absolute_path
from Scripts.text_cleaning import clean_text, bag_of_words
from nltk.stem import PorterStemmer
import random


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
num_ftrs = len(bag)

#Load in an untrained model:
model = NeuralNet(num_features, hidden_layer_1, hidden_layer_2, num_classes)

#Change randomised model parameters to trained params:
model.load_state_dict(trained_params)

# Set model to evaluation mode:
model.eval()

#Initialise Stemmer:
stem = PorterStemmer()

### Function to take in a message as text and output a response: 
def respond(message):
    clean_message = clean_text(message, stem)
    feature_vec = bag_of_words(clean_message, bag)
    # Reshape into a matrix
    feature_vec = feature_vec.reshape(1, num_ftrs)
    #Feed through model:
    output_vec = model(feature_vec)
    # Based on the fact Softmax function is an increasing function, the index of highest value is the class we're predicting,
    val, prediction = torch.max(output_vec, axis=1)
    # Note that the variable "prediction" is a label number, to get the actual label/tag we can use the label,mapping dictionary. 
    predicted_tag = list(label_mapping.keys())[prediction]
    # Only give a response if we're more than 75% sure that the tag is correct (ie the probabilitiy of this datapoint belonging to class is >= 0.75)
    probability = torch.softmax(val, axis=0)
    if probability >= 0.75:
        # Choose a random response from the predefined responses:
        for message_group in raw_data['messages']:
            if predicted_tag == message_group['tag']:
                random_response = random.choice(message_group['responses'])
                return random_response
    else:
        return "I'm not sure what you mean, please try a different message. :)"

        