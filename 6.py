import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Define Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Step 2: Define Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Step 3: Hyperparameters
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

img_size = 28
img_dim = img_size * img_size
noise_dim = 100
batch_size = 64
epochs = 20
learning_rate = 0.0002

# -------------------------------
# Step 4: Load MNIST Dataset
# -------------------------------
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

# -------------------------------
# Step 5: Initialize Models
# -------------------------------
generator = Generator(noise_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# -------------------------------
# Step 6: Training Loop
# -------------------------------
for epoch in range(epochs):
    for real_images, _ in dataloader:
        real_images = real_images.view(-1, img_dim).to(device)
        batch_size_curr = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        # ---- Train Discriminator ----
        noise = torch.randn(batch_size_curr, noise_dim).to(device)
        fake_images = generator(noise)

        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())

        loss_d = (criterion(real_output, real_labels) +
                  criterion(fake_output, fake_labels)) / 2

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # ---- Train Generator ----
        noise = torch.randn(batch_size_curr, noise_dim).to(device)
        fake_images = generator(noise)
        output = discriminator(fake_images)

        loss_g = criterion(output, real_labels)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_d:.4f} | Loss G: {loss_g:.4f}")

# -------------------------------
# Step 7: Generate Sample Images
# -------------------------------
noise = torch.randn(16, noise_dim).to(device)
generated_images = generator(noise).view(-1, 1, img_size, img_size).cpu()

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------------
# Step 8: Save Generator Model
# -------------------------------
torch.save(generator.state_dict(), "gan_generator.pth")
print("Generator model saved as gan_generator.pth")
