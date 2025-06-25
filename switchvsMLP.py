import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set device and AMP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler()

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# FLOP-Matched Width Calculation for Dense MLP
num_experts = 10  # used to compute flop_matched_dim for dense model
original_dim = 512
flop_matched_dim = int(original_dim / (num_experts ** 0.5))  # ≈162

# Dense (Standard) MLP model
class StandardMLP(nn.Module):
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, flop_matched_dim),
            nn.ReLU(),
            nn.Linear(flop_matched_dim, flop_matched_dim),
            nn.ReLU(),
            nn.Linear(flop_matched_dim, flop_matched_dim),
            nn.ReLU(),
            nn.Linear(flop_matched_dim, 10)
        )

    def forward(self, x):
        return self.model(x), None

# Switch Layer for Switch-based models
class SwitchLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.2):
        super(SwitchLayer, self).__init__()
        self.num_experts = num_experts
        self.routing_weights = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        self._init_weights()

    def _init_weights(self):
        s = 0.1
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    fan_in = layer.weight.size(1)
                    std = s / fan_in ** 0.5
                    nn.init.trunc_normal_(layer.weight, std=std)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_fp32 = x.to(torch.float32)
            routing_logits = self.routing_weights(x_fp32)
        expert_ids = torch.argmax(routing_logits, dim=1)
        expert_out_dim = self.experts[0][-1].out_features
        outputs = torch.zeros(x.size(0), expert_out_dim, device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            mask = (expert_ids == i)
            if mask.any():
                expert_out = self.experts[i](x[mask])
                outputs[mask] = expert_out
        return outputs, expert_ids

# MLP model that uses the SwitchLayer
class MLPWithSwitch(nn.Module):
    def __init__(self, num_experts):
        super(MLPWithSwitch, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, original_dim), nn.ReLU())
        self.switch = SwitchLayer(original_dim, original_dim, num_experts)
        self.layer3 = nn.Sequential(nn.ReLU(), nn.Linear(original_dim, 10))

    def forward(self, x):
        x = self.layer1(x)
        x, expert_ids = self.switch(x)
        x = self.layer3(x)
        return x, expert_ids

# Evaluation function calculates loss, accuracy, precision and recall (macro-averaged)
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda'):
                output, _ = model(x)
                loss = F.cross_entropy(output, y)
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return total_loss / total, acc, precision, recall

# Training function that records evaluation every 'eval_freq' epochs
def train_model(model, train_loader, test_loader, epochs=100, eval_freq=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {'epoch': [], 'test_loss': [], 'test_acc': [], 'time': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                output, _ = model(x)
                loss = F.cross_entropy(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)
        if (epoch + 1) % eval_freq == 0:
            epoch_time = time.time() - start_time
            test_loss, test_acc, _, _ = evaluate(model, test_loader)
            history['epoch'].append(epoch + 1)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['time'].append(epoch_time)
            print(f"Epoch {epoch+1:03d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s")
    return history

# Dictionary to hold histories for each model configuration
all_histories = {}
final_metrics = {}  # to store final evaluation metrics

# Train Dense (StandardMLP)
print("\nTraining Dense (StandardMLP) for 100 epochs")
dense_model = StandardMLP()
dense_history = train_model(dense_model, train_loader, test_loader, epochs=100, eval_freq=5)
all_histories['Dense'] = dense_history
_, dense_acc, dense_precision, dense_recall = evaluate(dense_model, test_loader)
final_metrics['Dense'] = {'acc': dense_acc, 'precision': dense_precision, 'recall': dense_recall}

# Train Switch models with 2, 4, 6, 8, and 10 experts
switch_expert_counts = [2, 4, 6, 8, 10]
for n in switch_expert_counts:
    print(f"\nTraining Switch MLP with {n} Experts for 100 epochs")
    switch_model = MLPWithSwitch(n)
    history = train_model(switch_model, train_loader, test_loader, epochs=100, eval_freq=5)
    all_histories[f'Switch_{n}'] = history
    _, switch_acc, switch_precision, switch_recall = evaluate(switch_model, test_loader)
    final_metrics[f'Switch_{n}'] = {'acc': switch_acc, 'precision': switch_precision, 'recall': switch_recall}

# Plotting combined graphs for Epoch Time, Test Accuracy, and Test Loss
colors = plt.cm.tab10.colors  # Get a set of colors

# Plot Epoch Time vs Epoch
plt.figure(figsize=(10, 6))
for idx, (name, history) in enumerate(all_histories.items()):
    plt.plot(history['epoch'], history['time'], label=name, color=colors[idx % len(colors)], marker='o')
plt.title('Epoch Time vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig('combined_epoch_time.png')
plt.show()

# Plot Test Accuracy vs Epoch
plt.figure(figsize=(10, 6))
for idx, (name, history) in enumerate(all_histories.items()):
    plt.plot(history['epoch'], history['test_acc'], label=name, color=colors[idx % len(colors)], marker='o')
plt.title('Test Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('combined_test_accuracy.png')
plt.show()

# Plot Test Loss vs Epoch
plt.figure(figsize=(10, 6))
for idx, (name, history) in enumerate(all_histories.items()):
    plt.plot(history['epoch'], history['test_loss'], label=name, color=colors[idx % len(colors)], marker='o')
plt.title('Test Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('combined_test_loss.png')
plt.show()

# Print final accuracy, precision, and recall for Dense and Switch with 10 Experts
print("\nFinal Metrics at 100th Epoch:")
print("Dense (StandardMLP):")
print(f"Accuracy: {final_metrics['Dense']['acc']:.4f}, Precision: {final_metrics['Dense']['precision']:.4f}, Recall: {final_metrics['Dense']['recall']:.4f}")
print("\nSwitch MLP with 10 Experts:")
print(f"Accuracy: {final_metrics['Switch_10']['acc']:.4f}, Precision: {final_metrics['Switch_10']['precision']:.4f}, Recall: {final_metrics['Switch_10']['recall']:.4f}")
