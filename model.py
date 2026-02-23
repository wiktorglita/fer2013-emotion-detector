import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)

        # 48 -> 24 -> 12 -> 6 (po 3 poolingach)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop(x)

        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x