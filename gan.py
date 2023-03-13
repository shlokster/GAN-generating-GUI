import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras import Model

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) 

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 3]))
    model.add(LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

# Define the loss function for the generator
def generator_loss(fake_output, input_images):
    return tf.losses.MeanSquaredError()(input_images, fake_output)

# Define the loss function for the discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define the optimizer for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the GAN model
def make_gan_model(generator, discriminator):
    gan = tf.keras.Sequential()
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    return gan

# Define the training loop
def train_gan(gan, generator, discriminator, dataset, epochs, batch_size):
    for epoch in range(epochs):
        for batch in dataset.batch(batch_size):
            # Generate fake images
            noise = tf.random.normal([batch_size, 100])
            generated_images = generator(noise)

            # Get real images
            real_images = batch

            # Train discriminator
            with tf.GradientTape() as tape:
                real_output = discriminator(real_images)
                fake_output = discriminator(generated_images)
                d_loss = discriminator_loss(real_output, fake_output)
            gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Train generator
            with tf.GradientTape() as tape:
                fake_output = discriminator(generated_images)
                g_loss = generator_loss(generated_images, fake_output)
            gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        # Print the losses
        print(f"Epoch {epoch+1}, Discriminator loss: {d_loss}, Generator loss: {g_loss}")

    # Generate new GUI design
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    return generated_image
