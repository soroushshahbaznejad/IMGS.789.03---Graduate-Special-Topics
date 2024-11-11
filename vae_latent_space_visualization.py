# Import necessary libraries (if not already imported)
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory to save plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

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

# Define the VAE model (with modification to return z)
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
        return recon_x, mu, log_var, z  # Return z

# Loss function
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD

# Initialize model, optimizer, and number of epochs
latent_dim = 20
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10  # Reduced for brevity

# Training loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var, _ = model(data)
        loss, _, _ = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# ==============================
# Latent Space Visualization
# ==============================

# Collect latent vectors and labels from test set
latent_vectors = []
labels_list = []

model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        recon_batch, mu, log_var, z = model(data)
        latent_vectors.append(z.cpu())
        labels_list.extend(labels.numpy())

# Concatenate all latent vectors
latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
labels_list = np.array(labels_list)

# Dimensionality reduction to 2D
# Option 1: t-SNE (uncomment to use t-SNE)
# tsne = TSNE(n_components=2, random_state=42)
# latent_2d = tsne.fit_transform(latent_vectors)

# Option 2: PCA
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_vectors)

# Plot the 2D scatter plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title('2D Visualization of VAE Latent Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'latent_space_visualization.png'))
plt.close()
