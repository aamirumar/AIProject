import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Set device and hyperparameters
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
batch_size = 32
epochs = 5            # reduced for faster execution
learning_rate = 0.001

# -------------------------------------------------
# Step 2: Data transformations
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Step 3: Load CIFAR-10 dataset
# -------------------------------------------------
train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------
# Step 4: Load pre-trained ResNet-18 model
# -------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -------------------------------------------------
# Step 5: Loss function and optimizer
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# -------------------------------------------------
# Step 6: Training function
# -------------------------------------------------
def train(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# -------------------------------------------------
# Step 7: Testing function
# -------------------------------------------------
def test(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# -------------------------------------------------
# Step 8: Train and evaluate
# -------------------------------------------------
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader)
    test_loss, test_acc = test(model, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f}, Test  Accuracy: {test_acc:.2f}%")

# -------------------------------------------------
# Step 9: Plot loss and accuracy
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy")
plt.legend()

plt.show()

# -------------------------------------------------
# Step 10: Save the trained model
# -------------------------------------------------
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Model saved as resnet18_cifar10.pth")
