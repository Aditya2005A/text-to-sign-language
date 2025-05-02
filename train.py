# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SignLanguageCNN
from preprocess_data import load_data
import os

def train_model(data_dir, batch_size=32, epochs=10, lr=0.001, save_path="checkpoints/model.pth"):
    # Load data
    (X_train, X_test, y_train, y_test), label_map = load_data(data_dir)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # NCHW
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageCNN(num_classes=len(label_map)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # line inside train.py
    train_model(data_dir="archive/train/asl_alphabet_train")