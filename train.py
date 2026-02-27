import json
from processing import tokenize, bag_of_words, stem
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


# ── Load intents ─────────────────────────────────────────────────────────────
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []  # (tokenized_pattern, tag) pairs

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

# Stem & deduplicate vocabulary
ignore_chars = ['?', '!', '.', ',', ';', ':', '-', '(', ')', '/']
all_words = [stem(w) for w in all_words if w not in ignore_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"✅ Loaded {len(tags)} intent tags, {len(all_words)} unique stemmed words, {len(xy)} training patterns")

# ── Build training data ───────────────────────────────────────────────────────
x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)


# ── Dataset & DataLoader ─────────────────────────────────────────────────────
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_sample = len(x_train)
        self.x_data = torch.tensor(x_train)
        self.y_data = torch.tensor(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample


# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_EPOCHS    = 1500       # more epochs for convergence on larger dataset
BATCH_SIZE    = 16         # larger batch for stability
LEARNING_RATE = 0.001
INPUT_SIZE    = len(all_words)
HIDDEN_SIZE   = 256        # much larger than original (was 8)
OUTPUT_SIZE   = len(tags)
LOG_INTERVAL  = 100        # print loss every N epochs

print(f"📐 Input size: {INPUT_SIZE} | Hidden size: {HIDDEN_SIZE} | Output (classes): {OUTPUT_SIZE}")

dataset = ChatDataSet()
train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ── Model, Loss, Optimizer ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Training on: {device}")

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, num_classes=OUTPUT_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler — reduces LR when loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

# ── Training Loop ─────────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss

    if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:>4}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.1f}%  LR: {lr:.6f}")

print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
FILE = "data.pth"
data = {
    "model_state": model.state_dict(),
    "input_size":  INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words":   all_words,
    "tags":        tags
}
torch.save(data, FILE)
print(f"💾 Model saved to '{FILE}'")
