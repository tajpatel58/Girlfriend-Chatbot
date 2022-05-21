# Girlfriend-Chatbot:

### Packages Used: 

- NLTK (Natural Language ToolKit) - Extracting Features from Messages data.
- Boto3 - Accessing Datasets on AWS. 
- PyTorch - Building Model.

### Project Overview: 

Creating a chatbot to speak to my girlfriend. Fun mini project to combine my NLP and PyTorch skills. The plan is to create a chat bot that will be trained on a messages dataset that I've written up where the datapoints are messages and the labels are categories like: "Greetings", "Goodbyes", "Day_Intents" etc. Then when a message comes in, the model/bot will predict which category the message belongs in and formulate a response based on some pre-determined responses for messages in this category.

### Step 1: Generating Dataset:
- Creating a dataset of common messages in a JSON format based on my previous conversations. 