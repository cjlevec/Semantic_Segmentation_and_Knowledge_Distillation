import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train = True

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

# Load PASCAL VOC 2012 dataset for training and validation
train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=image_transform, target_transform=target_transform)
val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=image_transform, target_transform=target_transform)

# Make instance of the fcn_resnet50 model (Pretrained on COCO VOC)
model = models.segmentation.fcn_resnet50(weights= 'DEFAULT', pretrained=True, progress=True)
# Need to modify the last layer of the classifier to output 21 classes for PASCAL
# (20 + 1 class for the background)
model.classifier[4] = torch.nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum= 0.9)

# Send model to GPU if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Device: {device}")

model = model.to(device)

# Initialize a list to store the loss values for each epoch
loss_values = []

# Training loop
if train == True:
  for epoch in range(50):
      model.train()
      running_loss = 0.0  # To calculate average loss for the epoch
      for inputs, labels in train_dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs['out'], labels.squeeze(1).long())
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      # Calculate average loss for the epoch
      epoch_loss = running_loss / len(train_dataloader)
      loss_values.append(epoch_loss)  # Append epoch loss to the list
      print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "weights.pth")

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), loss_values, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Function to compute IoU for a single batch
def calculate_iou(preds, targets, num_classes=21):
    preds = torch.argmax(preds, dim=1)  # Get predicted class labels
    iou_list = []

    for i in range(num_classes):
        intersection = torch.sum((preds == i) & (targets == i)).float()
        union = torch.sum((preds == i) | (targets == i)).float()
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        iou_list.append(iou.item())

    return iou_list

# Load weights file
model.load_state_dict(torch.load('./weights.pth', map_location='cpu'))  # Load trained weights


# Evaluation
model.eval()
# Tensor to hold mIoU for each class
iou_per_class = torch.zeros(21)
num_batches = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # Compute IoU for the current batch
        iou_batch = calculate_iou(outputs['out'], labels.squeeze(1).long(), num_classes=21)
        iou_per_class += torch.tensor(iou_batch)
        num_batches += 1

# Compute mean IoU over all classes
mean_iou = iou_per_class / num_batches
print(f"Mean IoU: {mean_iou.mean():.4f}")

# Optionally, you can also print per-class IoUs
for i, class_iou in enumerate(iou_per_class):
    print(f"Class {i} IoU: {class_iou:.4f}")