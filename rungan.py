import tensorflow as tf
import numpy as np
import os
import cv2
import /home/shlok/Codes/Project/350/gan.py

# Load the GUI dataset
gui_dataset = []
for file in os.listdir('gui_dataset'):
    img = cv2.imread(os.path.join('gui_dataset', file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28))
    gui_dataset.append(img)
gui_dataset = np.array(gui_dataset)

# Normalize the pixel values to [-1, 1]
gui_dataset = (gui_dataset.astype(np.float32) - 127.5) / 127.5 

# Define the GAN model and the generator and discriminator models
generator = make_generator_model()
discriminator = make_discriminator_model()
gan = make_gan_model(generator, discriminator)

# Train the GAN model
generated_image = train_gan(gan, generator, discriminator, gui_dataset, epochs=50, batch_size=32)

# Denormalize the generated image
generated_image = (generated_image.numpy()[0] * 127.5 + 127.5).astype(np.uint8)

# Save the generated image
cv2.imwrite('generated_gui.png', cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
