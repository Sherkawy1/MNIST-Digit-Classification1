import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Softmax Layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)

        return x

model = CNN()

criterion = nn.CrossEntropyLoss() #categorical entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    model.train()
    loss_sum = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            predicted = model(images).argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {loss_sum/len(train_loader):.4f} "
        f"Val Accuracy: {100*correct/total:.2f}%"
    )

model.eval()
correct = total = 0

with torch.no_grad():
    for images, labels in test_loader:
        predicted = model(images).argmax(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nTest Accuracy: {100*correct/total:.2f}%")

image, label = test_data[0]

with torch.no_grad():
    prediction = model(image.unsqueeze(0)).argmax(1).item()

print("\n--- Sample Prediction ---")
print(f"True Label: {label}")
print(f"Predicted Digit: {prediction}")
