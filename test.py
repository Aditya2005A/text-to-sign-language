# test.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SignLanguageCNN
from preprocess_data import load_data

def test_model(data_dir, model_path="checkpoints/model.pth"):
    # Load data
    (_, X_test, _, y_test), label_map = load_data(data_dir)

    # Convert to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)  # NCHW
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageCNN(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test loop
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    # line inside train.py
    test_model(data_dir="archive/test/asl_alphabet_test")