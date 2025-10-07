# model.py

import torch
import torch.nn as nn
from torchvision import models

class ResNet18Model(nn.Module):
    """
    Model menggunakan arsitektur ResNet18 dengan modifikasi:
    1. Layer pertama diubah untuk menerima input grayscale (1 channel)
    2. Layer terakhir diubah sesuai jumlah kelas
    3. Menggunakan pretrained weights dari ImageNet (opsional)
    """
    def __init__(self, in_channels, num_classes, pretrained=True):
        super(ResNet18Model, self).__init__()
        
        # Load ResNet18 pretrained model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modifikasi layer pertama untuk menerima in_channels (default ResNet18 = 3 channels)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        
        # Modifikasi layer terakhir (fc) sesuai dengan jumlah kelas
        num_features = self.resnet.fc.in_features
        if num_classes == 2:
            # Binary classification - output 1 neuron
            self.resnet.fc = nn.Linear(num_features, 1)
        else:
            # Multi-class classification
            self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """Mendefinisikan alur data (forward pass)."""
        return self.resnet(x)

# --- Bagian untuk pengujian ---
# Ganti nama model yang diuji dari SimpleCNN menjadi ResNet18Model
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'ResNet18Model' ---")
    
    model = ResNet18Model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28) # Uji dengan batch size 64
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}") # Harusnya [64, 1]
    print("Pengujian model 'ResNet18Model' berhasil.")