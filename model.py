import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, accuracy_score
import os

def plot_training_progress(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, accuracies, label='Training Accuracy', color='green')
    ax2.set_title('Training Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    }

    data_dir = os.path.join("..", "data")
              
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train']}
    
    dataset_size = len(image_datasets['train'])
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(image_datasets['train'], [train_size, val_size])

    dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
        }

    device = torch.device("cuda")

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.relu(out)
            return out

    class ResNet18(nn.Module):
        def __init__(self, num_classes=2):
            super(ResNet18, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, out_channels, blocks, stride=1):
            layers = []
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = ResNet18(num_classes=2)
    model.load_state_dict(torch.load('epoch_5.pth', weights_only=True))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 1

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()

    print('Training complete')

    plot_training_progress(train_losses, train_accuracies)

    # SAVE MODEL FOR LATER USE
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")

    # TESTING PHASE
    test = 1
    if test == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        # test data from https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset?resource=download
        test_dir = os.path.join("..", "data", "test")
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        #model = ResNet18(num_classes=2)

        #model.load_state_dict(torch.load('model.pth'))
        model.eval()
        all_labels = []
        all_preds = []

        correct_vehicle = 0
        correct_non_vehicle = 0 

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                correct_vehicle += torch.sum((preds == labels) & (labels == 1)).item()
                correct_non_vehicle += torch.sum((preds == labels) & (labels == 0)).item()


        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')

        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Precision: {precision:.4f}')

        print(f'Correct Vehicle Predictions: {correct_vehicle}')
        print(f'Correct Non-Vehicle Predictions: {correct_non_vehicle}')

if __name__ == '__main__':
    main()
