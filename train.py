import json
from processing import tokenize, bag_of_words, stem
# import random
# import chat
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


# import model

# open the json file and load its content
with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []  # tuple of tokenized word and the associated tag
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # checking for each single input of hte user
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        # collect the tokenized pattern in all_word list
        all_words.extend(word)
        xy.append((word, tag))

ignore_word = ['?', '!', '.', ',']
# stem all the words excluding punctuations
all_words = [stem(word) for word in all_words if word not in ignore_word]
# to remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

x_train = []  # for bag of words
y_train = []  # for indices
for (pattern_sentence, tag) in xy:
    bag =bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
num_epoch = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


# Example data preprocessing
def preprocess_data(x_train):
    # Convert X_train from list of lists of strings to float32 numpy array
    x_train = np.array([list(map(float, item)) for item in x_train], dtype=np.float32)
    return x_train


x_train = preprocess_data(x_train)


# defining Dataset and Dataloader


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_sample = len(x_train)
        self.x_data = torch.tensor(x_train)
        self.y_data = torch.tensor(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample


dataset = ChatDataSet()
# to occur process in main process and to not train the data any particular pattern
train_DataLoader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, num_classes=output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loop over epoch to train the model
for epoch in range(num_epoch):
    for (words, label) in train_DataLoader:
        words = words.to(device)
        label = label.to(dtype=torch.long).to(device)
        # forward pass
        outputs = model(words)
        loss = criterion(outputs, label)
        # zero the gradient before backword pass
        optimizer.zero_grad()
        loss.backward()
        # parameter update
        optimizer.step()

    # if (epoch+1) % 100 ==0:
    #     loss=None
    #     print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
print("Training Complete")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')

# bot_name = "Bots Health \U0001F468\u200D\u2695\ufe0f"
# print("Please enter your Symptoms along with ','....(press 'Bye' to exit!!)")
#
# while True:
#     sentence = input("You: ")
#     if sentence.lower() == "bye":
#         break
#
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])  # convert 1D BoW vector in 2D array
#     X = torch.from_numpy(X).to(device)
#
#     output = model(X)
#     # the maximum value in the output tensor
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]
#
#     probs = torch.softmax(output, dim=1)  # convert logits into probabilities
#     prob = probs[0][predicted.item()]  # probabilities for the first sample
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 if tag == "greeting" or tag == "thanks" or tag == "goodbye":
#                     print(f"{bot_name}: {intent['responses']}")
#                 else:
#                     print(f"{bot_name}: {"You might be facing "}{tag} {intent['responses']}")
#                 break
#     else:
#         print(f"{bot_name}: I do not understand...")
#
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)
