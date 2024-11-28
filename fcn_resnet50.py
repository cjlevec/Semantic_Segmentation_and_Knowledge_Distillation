import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from mpl_toolkits.mplot3d.proj3d import transform
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader


transform =transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load PASCAL VOC 2012 dataset
dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform)

# Make instance of the fcn_resnet50 model (Pretrained on COCO VOC)
model = models.segmentation.fcn_resnet50(weights= 'FCN_resnet50_weights.pth', pretrained=True, progress=True, num_classes=20)

# DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Send model to GPU if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = model.to(device)

# Training
for epoch in range (50):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")






