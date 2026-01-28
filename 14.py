import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Step 1: Device Configuration
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Step 2: Generator
# -------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Step 3: Discriminator
# -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Step 4: Hyperparameters
# -------------------------------------------------
noise_dim = 100
img_dim = 28 * 28
batch_size = 128          # increased for speed
lr = 0.0002
epochs = 10               # reduced (exam-safe)

# -------------------------------------------------
# Step 5: Load MNIST Dataset
# -------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------------------------
# Step 6: Initialize Models, Loss, Optimizers
# -------------------------------------------------
gen = Generator(noise_dim, img_dim).to(device)
disc = Discriminator(img_dim).to(device)

criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)

# -------------------------------------------------
# Step 7: Training Loop
# -------------------------------------------------
print("Starting GAN training...")
os.makedirs("generated_images", exist_ok=True)

for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.view(-1, img_dim).to(device)
        bs = real.size(0)

        # Labels
        real_labels = torch.ones(bs, 1).to(device)
        fake_labels = torch.zeros(bs, 1).to(device)

        # -------- Train Discriminator --------
        noise = torch.randn(bs, noise_dim).to(device)
        fake = gen(noise)

        loss_disc_real = criterion(disc(real), real_labels)
        loss_disc_fake = criterion(disc(fake.detach()), fake_labels)
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # -------- Train Generator --------
        noise = torch.randn(bs, noise_dim).to(device)
        fake = gen(noise)
        loss_gen = criterion(disc(fake), real_labels)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_disc:.4f} | Loss G: {loss_gen:.4f}")

    # -------------------------------------------------
    # Save sample generated images
    # -------------------------------------------------
    with torch.no_grad():
        sample_noise = torch.randn(16, noise_dim).to(device)
        fake_images = gen(sample_noise).view(-1, 1, 28, 28).cpu()

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i][0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
plt.savefig(f"generated_images/epoch_{epoch+1}.png")
plt.close()



print("Training complete. Images saved in 'generated_images/' folder.")
