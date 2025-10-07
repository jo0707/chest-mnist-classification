# model.py

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Arsitektur CNN yang paling sederhana dan basic:
    1. Dua lapisan konvolusi
    2. MaxPooling setelah setiap konvolusi
    3. Satu fully connected layer untuk output
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)   # 28x28 → 28x28
        self.pool = nn.AvgPool2d(2)                                                  # 28x28 → 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)                                 # 14x14 → 10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                                        # 10x10 → 5x5 setelah pool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # (N, 6, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))   # (N,16, 5, 5)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'SimpleCNN' ---")
    
    model = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'SimpleCNN' berhasil.")