import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split # divide the dataset
import torch.nn as nn
import torch.optim as optim

transform = transforms.ToTensor()

full_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(f"Original dataset size: {len(full_dataset)}")

train_size = int(0.80 * len(full_dataset))
validation_size = int(0.10 * len(full_dataset))
test_size = len(full_dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, validation_size, test_size],
    generator=torch.Generator().manual_seed(42) # to make the split random
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNet()

criterion = nn.CrossEntropyLoss() # to calculate the error bet. prediction and label
optimizer = optim.Adam(model.parameters(), lr=0.001) #to update the gradients

epochs = 4

print("\nStarting training...")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            outputs = model(images) #making prediction

            _, predicted = torch.max(outputs.data, 1) #choose the highest prediction

            total += labels.size(0)

            correct += (predicted == labels).sum().item() #to compare how many of them is correct

    validation_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

print("\nTraining finished successfully!")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

with torch.no_grad():

    sample_image, sample_label = test_dataset[0]

    output = model(sample_image.unsqueeze(0))

    predicted_digit = torch.argmax(output, dim=1).item()

print("\n--- Sample Prediction ---")
print(f"True Label: {sample_label}")
print(f"Predicted Digit: {predicted_digit}")