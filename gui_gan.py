# To run the above code, you would need to have the following dependencies installed:

#     PyTorch
#     torchvision
#     numpy
#     matplotlib

# Here are the general steps to follow:

#     Save the code in a file, say gui_gan.py.
#     Replace 'path/to/dataset' with the actual path to your dataset of GUI images.
#     Create a directory named generated_images in the same directory as gui_gan.py (this is where the generated images will be saved).
#     Open a terminal/command prompt in the directory containing gui_gan.py.
#     Run the command python gui_gan.py to start training the GAN.

# The generated images will be saved in the generated_images directory. Once the training is complete, the trained generator will be saved in a file named generator.pth. You can use this file to generate new GUI images using the generate_images.py script provided earlier.    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from PIL import Image
import os

# Define custom dataset for loading web GUI images
class WebGUIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Define generator network
class Generator(nn.Module):
    def __init__(self, latent_size=100, ngf=64):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.ngf = ngf

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, self.ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Define discriminator network
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.main = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# Initialize generator and discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Configure data loader
batch_size = 64
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = WebGUIDataset(root_dir='/home/shlok/Codes/Project/350/screenshots', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Train the GAN
num_epochs = 200
for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        # Train discriminator
        optimizer_D.zero_grad()
        label_real = torch.full((batch_size,), 1, device=device)
        label_fake = torch.full((batch_size,), 0, device=device)
        real_images = real_images.to(device)
        fake_images = generator(torch.randn(batch_size, 100, 1, 1, device=device))
        output_real = discriminator(real_images).view(-1)
        output_fake = discriminator(fake_images.detach()).view(-1)
        loss_D_real = adversarial_loss(output_real, label_real)
        loss_D_fake = adversarial_loss(output_fake, label_fake)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_images).view(-1)
        loss_G = adversarial_loss(output_fake, label_real)
        loss_G.backward()
        optimizer_G.step()

        # Print losses
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save generated images
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(batch_size, 100, 1, 1, device=device))
        save_image(fake_images, f"generated_images/{epoch}.png", normalize=True)

# Save trained generator
torch.save(generator.state_dict(), 'path/to/generator.pth')



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
# from torchvision.utils import save_image
# from PIL import Image
# import os

# # Define custom dataset for loading web GUI images
# class WebGUIDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = self.image_files[idx]
#         image = Image.open(image_path)
#         if self.transform:
#             image = self.transform(image)
#         return image

# # Define generator network
# class Generator(nn.Module):
#     def __init__(self, latent_size=100, ngf=64):
#         super(Generator, self).__init__()
#         self.latent_size = latent_size
#         self.ngf = ngf

#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(self.latent_size, self.ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(self.ngf*8),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ngf*4),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ngf*2),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ngf),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(self.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.main(x)

# class Discriminator(nn.Module):
#     def __init__(self, ndf=64):
#         super(Discriminator, self).__init__()
#         self.ndf = ndf

#         self.main = nn.Sequential(
#             nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ndf*4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(self.ndf*8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.main(x)

# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define hyperparameters
# batch_size = 64
# latent_size = 100
# num_epochs = 50
# learning_rate = 0.0002
# beta1 = 0.5

# # Define transforms for data augmentation and normalization
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load dataset
# webgui_dataset = WebGUIDataset('path/to/gui/images', transform=transform)
# dataloader = DataLoader(webgui_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# # Initialize generator and discriminator
# generator = Generator(latent_size=latent_size).to(device)
# discriminator = Discriminator().to(device)

# # Define loss function and optimizers
# adversarial_loss = nn.BCELoss()
# optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# # Train GAN
# for epoch in range(num_epochs):
#     for i, data in enumerate(dataloader):
#         real_images = data.to(device)
#         batch_size = real_images.size(0)

#         # Train discriminator with real images
#         label_real = torch.full((batch_size,), 1, device=device)
#         output_real = discriminator(real_images).view(-1)
#         loss_D_real = adversarial_loss(output_real, label_real)
#         loss_D_real.backward()

#         # Train discriminator with fake images
#         noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
#         fake_images = generator(noise)
#         label_fake = torch.full((batch_size,), 0, device=device)
#         output_fake = discriminator(fake_images.detach()).view(-1)
#         loss_D_fake = adversarial_loss(output_fake, label_fake)
#         loss_D_fake.backward()
