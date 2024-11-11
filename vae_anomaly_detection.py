# Import necessary libraries
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
from sklearn.metrics import classification_report, roc_curve, auc

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

# Define the VAE model
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
num_epochs = 50

# Lists to keep track of losses
train_loss_list = []
bce_loss_list = []
kld_loss_list = []

# Training loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
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
    test_data = test_data.to(device)
    recon_batch, _, _ = model(test_data)
    n = 8  # Number of images to display
    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_data[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')
        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')
    # Save the figure
    plt.savefig(os.path.join(plot_dir, 'reconstructed_images.png'))
    plt.close()

# Generate new images by sampling from the latent space
with torch.no_grad():
    # Sample random vectors from the latent space
    z = torch.randn(64, latent_dim).to(device)
    sample = model.decode(z).cpu()
    plt.figure(figsize=(8, 8))
    for i in range(64):
        ax = plt.subplot(8, 8, i + 1)
        plt.imshow(sample[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # Save the figure
    plt.savefig(os.path.join(plot_dir, 'generated_images.png'))
    plt.close()

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
# Save the figure
plt.savefig(os.path.join(plot_dir, 'training_loss.png'))
plt.close()

# ==============================
# Anomaly Detection with VAE
# ==============================

# Function to add noise to images
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

# Get a batch of test images
test_loader_iter = iter(test_loader)
test_images, _ = next(test_loader_iter)
test_images = test_images.to(device)

# Create anomalous images
anomalous_images = add_noise(test_images)

# Function to compute reconstruction error
def compute_reconstruction_error(model, images):
    model.eval()
    with torch.no_grad():
        recon_images, _, _ = model(images)
        recon_images = recon_images.view(-1, 1, 28, 28)
        errors = torch.sum((recon_images - images) ** 2, dim=[1, 2, 3])
    return errors.cpu().numpy()

# Compute errors for normal images
normal_errors = compute_reconstruction_error(model, test_images)

# Compute errors for anomalous images
anomalous_errors = compute_reconstruction_error(model, anomalous_images)

# Plot the distribution of reconstruction errors
plt.figure(figsize=(10, 5))
sns.histplot(normal_errors, bins=50, color='blue', label='Normal', stat='density', kde=True)
sns.histplot(anomalous_errors, bins=50, color='red', label='Anomalous', stat='density', kde=True)
plt.title('Distribution of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'reconstruction_error_distribution.png'))
plt.close()

# Set a threshold for anomaly detection
mean_error = np.mean(normal_errors)
std_error = np.std(normal_errors)
threshold = mean_error + 3 * std_error  # 3 standard deviations from the mean
print(f"Threshold for anomaly detection: {threshold:.4f}")

# Create labels: 0 for normal, 1 for anomalous
normal_labels = np.zeros(len(normal_errors))
anomalous_labels = np.ones(len(anomalous_errors))

# Concatenate errors and labels
all_errors = np.concatenate([normal_errors, anomalous_errors])
all_labels = np.concatenate([normal_labels, anomalous_labels])

# Classify based on threshold
predictions = (all_errors > threshold).astype(int)

# Print classification report
print(classification_report(all_labels, predictions, target_names=['Normal', 'Anomalous']))

# Compute ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_errors)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
plt.close()

# Visualize some anomalous reconstructions
# Select a few anomalous images
num_images = 8
sample_anomalous_images = anomalous_images[:num_images]
sample_reconstructions, _, _ = model(sample_anomalous_images)

# Plot original and reconstructed anomalous images
plt.figure(figsize=(15, 5))
for i in range(num_images):
    # Original anomalous images
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(sample_anomalous_images[i].cpu().detach().view(28, 28), cmap='gray')
    ax.axis('off')
    # Reconstructed images
    ax = plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(sample_reconstructions[i].cpu().detach().view(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle('Anomalous Images and Their Reconstructions')
plt.savefig(os.path.join(plot_dir, 'anomalous_reconstructions.png'))
plt.close()
