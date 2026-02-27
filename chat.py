import torch
from processing import tokenize, bag_of_words
# import model
import json

# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# import model
from model import NeuralNet


class ChatBot:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bot_name = "Bots Health \U0001F468\u200D\u2695\ufe0f"
        print("Please enter your Symptoms along with ','....(press 'Bye' to exit!!)")

        with open(r'intents.json', 'r', encoding='utf-8') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE, weights_only=True)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def take_response(self, sentence):
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])  # convert 1D BoW vector in 2D array
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        # the maximum value in the output tensor
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)  # convert logits into probabilities
        prob = probs[0][predicted.item()]  # probabilities for the first sample
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    if tag == "greeting" or tag == "thanks" or tag == "goodbye":
                        return f"{self.bot_name}: {intent['responses']}"
                    else:
                        prefix = "You might be facing "
                        return f"{self.bot_name}: {prefix}{tag} {intent['responses']}"
                    break
        else:
            return f"{self.bot_name}: I do not understand..."

