import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import sys
from torch.utils.data import random_split, DataLoader
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Add project root to path for custom module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Import custom TaskDataset
from dataset import TaskDataset

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.11386307, 0.11385383, 0.11392628], std=[0.11869919, 0.11869682, 0.11888567])
])

# Load dataset (relative to project root)
data_path = os.path.join(project_root, 'data', 'Train.pt')
try:
    dataset = torch.load(data_path, weights_only=False)
    if not isinstance(dataset, TaskDataset):
        raise ValueError("Loaded dataset is not a TaskDataset")
    dataset.transform = transform
    dataset.imgs = [img.convert('RGB') for img in dataset.imgs]
    print("Successfully loaded Train.pt")
except Exception as e:
    print(f"Error loading Train.pt: {e}")
    exit()

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders (GPU-optimized)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

# Initialize model with pre-trained weights
model = models.resnet34(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Adversarial attack functions
def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images

def pgd_attack(model, images, labels, eps, alpha, steps=10):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_images = images.clone().detach()
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad
        adv_images = adv_images + alpha * grad.sign()
        adv_images = torch.clamp(adv_images, images - eps, images + eps)
        adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images

def evaluate_adversarial(model, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0
    for _, images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = images.clone().detach()
        for _ in range(10):  # 10-step PGD for evaluation
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            grad = adv_images.grad
            adv_images = adv_images + (2 / 255) * grad.sign()
            adv_images = torch.clamp(adv_images, images - epsilon, images + epsilon)
            adv_images = torch.clamp(adv_images, 0, 1)
        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# Adversarial training
num_epochs = 100
epsilon = 2 / 255
alpha = 2 / 255
beta = 1.0  # Clean-only loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (_, images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Create tensor for epsilon
        eps_tensor = torch.full((batch_size,), epsilon, device=device)

        # Clean data pass
        optimizer.zero_grad()
        outputs = model(images)
        loss_clean = criterion(outputs, labels)

        # Generate adversarial examples (mix of FGSM and PGD)
        if np.random.rand() < 0.5:
            adv_images = fgsm_attack(model, images, labels, epsilon)
        else:
            adv_images = pgd_attack(model, images, labels, eps_tensor, alpha)
        
        # Adversarial data pass
        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        # Combined loss
        loss = beta * loss_clean + (1 - beta) * loss_adv
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    scheduler.step()

    # Evaluate on test set
    model.eval()
    clean_acc = evaluate_adversarial(model, test_loader, torch.zeros(128).to(device))
    adv_acc = evaluate_adversarial(model, test_loader, torch.ones(128).to(device) * epsilon)
    print(f'[Epoch {epoch + 1}] Clean Acc: {clean_acc:.2f}%, Adv Acc: {adv_acc:.2f}%')

# Save model (relative to project root)
model_out_dir = os.path.join(project_root, 'out', 'models')
os.makedirs(model_out_dir, exist_ok=True)
model_path = os.path.join(model_out_dir, 'robust_resnet34.pt')
torch.save(model.state_dict(), model_path)

# Test model compliance
allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
with open(model_path, "rb") as f:
    try:
        test_model = allowed_models["resnet34"](weights=None)
        test_model.fc = nn.Linear(test_model.fc.in_features, 10)
        state_dict = torch.load(f, map_location=torch.device("cpu"))
        test_model.load_state_dict(state_dict, strict=True)
        test_model.eval()
        out = test_model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10), "Invalid output shape"
        print("Model passed local assertions")
    except Exception as e:
        print(f"Assertion failed: {e}")

# Submission code
import requests
response = requests.post(
    "http://34.122.51.94:9090/robustness",
    files={"file": open(model_path, "rb")},
    headers={"token": "34811541", "model-name": "resnet34"}
)
print(response.json())
