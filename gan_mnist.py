import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ========================
# 1. Load the MNIST Dataset
# ========================

# Transformation: Convert images to tensors and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

batch_size = 100

# Load the dataset
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# ========================
# 2. Define the Generator Network
# ========================

class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=784):  # 28x28 images
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size),
            nn.Tanh(),  # Output values between -1 and 1
        )

    def forward(self, x):
        return self.model(x)

# ========================
# 3. Define the Discriminator Network
# ========================

class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

# ========================
# 4. Initialize Networks, Loss Function, and Optimizers
# ========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the networks
G = Generator().to(device)
D = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
lr = 0.0002
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Ensure the output directory exists
os.makedirs('generated_images', exist_ok=True)

# ========================
# 5. Training Loop
# ========================

num_epochs = 100
G_losses = []
D_losses = []
img_list = []
fixed_noise = torch.randn(64, 100, device=device)  # For consistent image generation

for epoch in range(1, num_epochs + 1):
    for i, (real_images, _) in enumerate(train_loader):
        # Prepare real images
        real_images = real_images.view(-1, 784).to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        D.zero_grad()
        # Loss for real images
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        # Loss for fake images
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = G(noise)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        # Backpropagation and optimization
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        # Train Generator
        # -----------------
        G.zero_grad()
        # Generator wants discriminator to believe generated images are real
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        # Backpropagation and optimization
        g_loss.backward()
        optimizer_G.step()
        
    # Record losses
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())
    
    # Save generated images at specified epochs
    if epoch in [1, 10, 50, 100]:
        with torch.no_grad():
            fake_images = G(fixed_noise).reshape(-1, 1, 28, 28)
            img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
            img_list.append(img_grid)
            # Save the generated images
            vutils.save_image(fake_images, f'generated_images/generated_epoch_{epoch}.png', nrow=8, normalize=True)
        print(f'Epoch [{epoch}/{num_epochs}]  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')

# ========================
# 6. Plot Loss Curves
# ========================

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(range(1, num_epochs+1), G_losses, label="Generator")
plt.plot(range(1, num_epochs+1), D_losses, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('generated_images/loss_curves.png')
plt.show()

# ========================
# 7. Visualize Generated Images
# ========================

def show_saved_images(epochs):
    for epoch in epochs:
        img_path = f'generated_images/generated_epoch_{epoch}.png'
        image = mpimg.imread(img_path)
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title(f'Generated Images at Epoch {epoch}')
        plt.imshow(image)
        plt.show()

# Display the images for epochs 1, 10, 50, and 100
show_saved_images([1, 10, 50, 100])
