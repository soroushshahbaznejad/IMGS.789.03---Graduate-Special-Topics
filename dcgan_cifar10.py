import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================
# 1. Hyperparameters and Configuration
# ========================

# Set random seed for reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataroot = 'data'
batch_size = 128
image_size = 64  # Resize CIFAR-10 images to 64x64
nc = 3          # Number of channels in the training images (CIFAR-10 is RGB)
nz = 100        # Size of z latent vector (i.e., size of generator input)
ngf = 64        # Size of feature maps in generator
ndf = 64        # Size of feature maps in discriminator
num_epochs = 25
lr = 0.0002
beta1 = 0.5     # Beta1 hyperparam for Adam optimizers

# Create output directory
os.makedirs('dcgan_results', exist_ok=True)

# ========================
# 2. Data Loading and Preprocessing
# ========================

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

# Load the dataset
dataset = torchvision.datasets.CIFAR10(root=dataroot, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# ========================
# 3. Define the Generator Network
# ========================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # (nz) -> (ngf*8 x 4 x 4)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8 x 4 x 4) -> (ngf*4 x 8 x 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # stride=2, padding=1
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4 x 8 x 8) -> (ngf*2 x 16 x 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2 x 16 x 16) -> (ngf x 32 x 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf x 32 x 32) -> (nc x 64 x 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output values between [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the generator
netG = Generator().to(device)

# ========================
# 4. Define the Discriminator Network
# ========================

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc x 64 x 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # (nc) -> (ndf x 32 x 32)
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf x 32 x 32) -> (ndf*2 x 16 x 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # stride=2, padding=1
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2 x 16 x 16) -> (ndf*4 x 8 x 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4 x 8 x 8) -> (ndf*8 x 4 x 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8 x 4 x 4) -> (1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, input):
        return self.main(input).view(-1)

# Instantiate the discriminator
netD = Discriminator().to(device)

# ========================
# 5. Loss Function and Optimizers
# ========================

criterion = nn.BCELoss()

# Create batch of latent vectors for visualization
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# ========================
# 6. Training Loop
# ========================

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=device)  # Real label = 1

        output = netD(real_images)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(0.)  # Fake label = 0

        output = netD(fake_images.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        label.fill_(1.)  # Generator wants discriminator to believe fake images are real
        output = netD(fake_images)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {errD.item():.4f}, Loss G: {errG.item():.4f} \
                  D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving its output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iters += 1

# ========================
# 7. Visualize the Results
# ========================

# Import necessary modules
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Plot the losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('dcgan_results/loss_curves.png')
plt.show()

# Animation showing the improvements of the generator
fig = plt.figure(figsize=(8,8))
plt.axis("off")

ims = []
for img in img_list:
    ims.append([plt.imshow(np.transpose(img, (1,2,0)), animated=True)])

ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)

# Save the animation as a GIF file
ani.save('dcgan_results/generation_animation.gif', writer='pillow')
print("Animation saved as 'dcgan_results/generation_animation.gif'.")

# Display real images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(
    vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.savefig('dcgan_results/real_images.png')
plt.show()

# Display generated images from the last epoch
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1,2,0)))
plt.savefig('dcgan_results/fake_images_epoch_%d.png' % num_epochs)
plt.show()
# ========================
# 8. Save the Trained Generator Model
# ========================

# Save the trained generator model
torch.save(netG.state_dict(), 'generator.pth')
print("Generator model saved as 'generator.pth'.")
