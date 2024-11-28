import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader


# Define transforms for image and target separately
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.PILToTensor(),
    transforms.Lambda(lambda t: torch.clamp(t, 0, 20).long())  # Clamp maps pixels part of background class (class 255) to 20
])

# Load PASCAL VOC 2012 dataset
dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=image_transform, target_transform= target_transform)

# Make instance of the fcn_resnet50 model (Pretrained on COCO VOC)
model = models.segmentation.fcn_resnet50(weights= 'DEFAULT', pretrained=True, progress=True)
# Need to modify the last layer of the classifier to output 21 classes for PASCAL
# (20 + 1 class for the background)
model.classifier[4] = torch.nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

# DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Send model to GPU if available
device = 'cpu'
"""if torch.cuda.is_available():
    device = 'cuda'"""

model = model.to(device)

# Training
for epoch in range (50):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs['out'], labels.squeeze(1).long())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")






