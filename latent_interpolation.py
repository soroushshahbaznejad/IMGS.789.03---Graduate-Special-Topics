import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================================
# 1. Set Up Configuration and Parameters
# ========================================

# Ensure reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator parameters (must match the trained model)
nz = 100  # Size of the latent vector (input to generator)
ngf = 64  # Size of feature maps in generator
nc = 3    # Number of channels in the generated images (CIFAR-10 is RGB)

# Output directory
os.makedirs('latent_interpolation_results', exist_ok=True)

# ========================================
# 2. Define the Generator Network
# ========================================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z latent vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # Output: (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # Output: (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # Output: (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # Output: (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # Output: (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ========================================
# 3. Load the Pre-trained Generator
# ========================================

# Instantiate the generator
netG = Generator().to(device)

# Load pre-trained weights
# Replace 'generator.pth' with the path to your saved generator model
if os.path.exists('generator.pth'):
    netG.load_state_dict(torch.load('generator.pth', map_location=device))
    netG.eval()  # Set the generator to evaluation mode
    print("Pre-trained generator model loaded.")
else:
    print("Error: Pre-trained generator model 'generator.pth' not found.")
    print("Please train the DCGAN model first and save the generator's state dict as 'generator.pth'.")
    exit()

# ========================================
# 4. Select Two Latent Vectors
# ========================================

# Generate two random latent vectors
z_dim = nz  # Should match the generator's input size
z1 = torch.randn(1, z_dim, 1, 1, device=device)
z2 = torch.randn(1, z_dim, 1, 1, device=device)

# ========================================
# 5. Interpolate Between the Vectors
# ========================================

# Number of interpolation steps
num_interpolations = 10

# Linearly interpolate between z1 and z2
interpolated_z = []
alphas = np.linspace(0, 1, num_interpolations)
for alpha in alphas:
    z = (1 - alpha) * z1 + alpha * z2
    interpolated_z.append(z)

# ========================================
# 6. Generate Images from Interpolated Vectors
# ========================================

# Generate images
generated_images = []
with torch.no_grad():
    for z in interpolated_z:
        fake_image = netG(z).cpu()
        generated_images.append(fake_image)

# ========================================
# 7. Visualize the Resulting Images
# ========================================

# Prepare images for display
grid = vutils.make_grid(torch.cat(generated_images, dim=0), nrow=num_interpolations, padding=2, normalize=True)

# Save the grid image
vutils.save_image(grid, 'latent_interpolation_results/interpolation.png', normalize=False)
print("Interpolation image saved to 'latent_interpolation_results/interpolation.png'.")

# Plot the images
plt.figure(figsize=(20, 5))
plt.axis('off')
plt.title('Latent Space Interpolation')
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()

# ========================================
# 8. Comment on the Transitions
# ========================================

print("The interpolation shows a smooth transition between two generated images.")
print("Observe how the features in the images gradually change from one to the other.")
