# Girlfriend-Chatbot:

### Packages Used: 

- NLTK (Natural Language ToolKit) - Pre-Processing/Extracting Features from Messages data.
- Boto3 - Accessing Datasets on AWS. 
- PyTorch - Building Model.

### Project Overview: 

Creating a chatbot to speak to my girlfriend. Fun mini project to combine my NLP and PyTorch skills. The plan is to create a chat bot that will be trained on a messages dataset that I've written up where the datapoints are messages and the labels are categories like: "Greetings", "Goodbyes", "Day_Intents" etc. Then when a message comes in, the model/bot will predict which category the message belongs in and formulate a response based on some pre-determined responses for messages in this category.

### Step 1: Generating Dataset:
- Creating a dataset of common messages in a JSON format based on my previous conversations. 

### Step 2: Data Preprocessing:
- I'll apply a typical NLP approach in extracting features. 
    1. Lowercase all characters
    2. Remove combinations of "x"s and "o"s at the end of messages. As they hold no information as to which category messages belong in. 
    3. Tokenize
    4. Apply Lematization or Stemming, most likely the latter, as often in messages words are extended for no reason, eg: "Goooood Morninggg".


### Step 3: Building the Model:
- I'll be using a relatively simple Neural Network initially and depending on model/chatbot performance, I can ammend. 
- The net will be formed with 2 hidden layers, first with 10 units and the second with 6 units. 

### Step 4: Training Loop: 
- Running the training loop by passing in data in batches of 6. 
- I'll store the model parameters for the model with the best test accuracy. 
- Using the Adam Optimizer.