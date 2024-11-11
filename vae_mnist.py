import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define transformations for the training data and testing data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the training and test datasets
train_dataset = datasets.MNIST(root='mnist_data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='mnist_data', train=False,
                              transform=transform, download=True)

# Data loaders
batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False)

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean of latent space
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance of latent space

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD

# Initialize model, optimizer, and number of epochs
latent_dim = 20
model = VAE(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20

# Lists to keep track of losses
train_loss_list = []
bce_loss_list = []
kld_loss_list = []

model.train()
for epoch in range(num_epochs):
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = bce_loss / len(train_loader.dataset)
    avg_kld = kld_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    bce_loss_list.append(avg_bce)
    kld_loss_list.append(avg_kld)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}')

# Visualize reconstructed images
model.eval()
with torch.no_grad():
    # Get a batch of test data
    test_data, _ = next(iter(test_loader))
    recon_batch, _, _ = model(test_data)
    n = 8  # Number of images to display
    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_data[i].view(28, 28), cmap='gray')
        ax.axis('off')
        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_batch[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

# Generate new images by sampling from the latent space
with torch.no_grad():
    # Sample random vectors from the latent space
    z = torch.randn(64, latent_dim)
    sample = model.decode(z).cpu()
    plt.figure(figsize=(8, 8))
    for i in range(64):
        ax = plt.subplot(8, 8, i + 1)
        plt.imshow(sample[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

# Plot ELBO Loss and KL-Divergence
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

# Plot ELBO Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, label='ELBO Loss')
plt.title('ELBO Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot KL-Divergence
plt.subplot(1, 2, 2)
plt.plot(epochs, kld_loss_list, label='KL Divergence', color='orange')
plt.title('KL Divergence Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.legend()

plt.tight_layout()
plt.show()
