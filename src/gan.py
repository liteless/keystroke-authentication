import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List


class VAEEncoder(keras.Model):
    """
    Encoder: Maps digraph sequences to latent distribution (mean, log_var)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: List[int] = [64, 128, 256],
        dropout: float = 0.3,
        **kwargs
    ):
        super(VAEEncoder, self).__init__(**kwargs)

        self.latent_dim = latent_dim

        # Convolutional layers to extract features
        self.conv1 = layers.Conv1D(
            hidden_dims[0], 5, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
            hidden_dims[1], 5, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout)

        self.conv3 = layers.Conv1D(
            hidden_dims[2], 3, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()

        # Global pooling to handle variable sequence lengths
        self.global_pool = layers.GlobalAveragePooling1D()

        # Latent distribution parameters
        self.fc_mean = layers.Dense(latent_dim, name='z_mean')
        self.fc_logvar = layers.Dense(latent_dim, name='z_log_var')

    def call(self, x, training=None):
        """
        Input: (batch_size, seq_len, 9)
        Output: (z_mean, z_log_var) each of shape (batch_size, latent_dim)
        """
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = self.dropout1(h, training=training)

        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.dropout2(h, training=training)

        h = self.conv3(h)
        h = self.bn3(h, training=training)

        h = self.global_pool(h)

        z_mean = self.fc_mean(h)
        z_log_var = self.fc_logvar(h)

        return z_mean, z_log_var


class VAEDecoder(keras.Model):
    """
    Decoder: Maps latent code to digraph sequence
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_seq_len: int = 50,
        output_dim: int = 9,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        **kwargs
    ):
        super(VAEDecoder, self).__init__(**kwargs)

        self.output_seq_len = output_seq_len
        self.output_dim = output_dim

        # Project latent code to initial sequence
        self.fc = layers.Dense(
            output_seq_len * hidden_dims[0], activation='relu')
        self.reshape = layers.Reshape((output_seq_len, hidden_dims[0]))

        # Transposed convolutions to upsample
        self.conv1 = layers.Conv1D(
            hidden_dims[0], 5, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
            hidden_dims[1], 5, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout)

        self.conv3 = layers.Conv1D(
            hidden_dims[2], 3, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()

        # Output layer
        self.output_layer = layers.Conv1D(output_dim, 1, padding='same')

    def call(self, z, training=None):
        """
        Input: (batch_size, latent_dim)
        Output: (batch_size, seq_len, 9)
        """
        h = self.fc(z)
        h = self.reshape(h)

        h = self.conv1(h)
        h = self.bn1(h, training=training)
        h = self.dropout1(h, training=training)

        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.dropout2(h, training=training)

        h = self.conv3(h)
        h = self.bn3(h, training=training)

        output = self.output_layer(h)

        return output


class Discriminator(keras.Model):
    """
    Discriminator: Classifies real vs fake digraph sequences
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dims: List[int] = [64, 128, 256],
        dropout: float = 0.3,
        **kwargs
    ):
        super(Discriminator, self).__init__(**kwargs)

        self.conv1 = layers.Conv1D(
            hidden_dims[0], 5, strides=2, padding='same')
        self.leaky1 = layers.LeakyReLU(0.2)
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
            hidden_dims[1], 5, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.leaky2 = layers.LeakyReLU(0.2)
        self.dropout2 = layers.Dropout(dropout)

        self.conv3 = layers.Conv1D(
            hidden_dims[2], 3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.leaky3 = layers.LeakyReLU(0.2)

        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(1)  # Real/fake logit

    def call(self, x, training=None):
        """
        Input: (batch_size, seq_len, 9)
        Output: (batch_size, 1) - logit for real/fake
        """
        h = self.conv1(x)
        h = self.leaky1(h)
        h = self.dropout1(h, training=training)

        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.leaky2(h)
        h = self.dropout2(h, training=training)

        h = self.conv3(h)
        h = self.bn3(h, training=training)
        h = self.leaky3(h)

        h = self.global_pool(h)
        logit = self.fc(h)

        return logit


class VAEGAN(keras.Model):
    """
    VAE-GAN for keystroke data generation
    Combines VAE reconstruction with GAN adversarial training
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 64,
        seq_len: int = 50,
        **kwargs
    ):
        super(VAEGAN, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(
            latent_dim=latent_dim, output_seq_len=seq_len, output_dim=input_dim)
        self.discriminator = Discriminator(input_dim=input_dim)

        # Separate optimizers for each component
        self.vae_optimizer = None
        self.disc_optimizer = None

        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.vae_gan_loss_tracker = keras.metrics.Mean(name="vae_gan_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.vae_gan_loss_tracker,
            self.disc_loss_tracker,
        ]

    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick"""
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=None):
        """
        Forward pass through VAE
        """
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        return reconstructed, z_mean, z_log_var

    def compile(self, optimizer, **kwargs):
        """Override compile to set up optimizers"""
        super(VAEGAN, self).compile(**kwargs)
        # Create separate optimizers with same config
        if isinstance(optimizer, str):
            self.vae_optimizer = keras.optimizers.get(optimizer)
            self.disc_optimizer = keras.optimizers.get(optimizer)
        else:
            # Clone the optimizer config
            config = optimizer.get_config()
            self.vae_optimizer = optimizer.__class__.from_config(config)
            self.disc_optimizer = optimizer.__class__.from_config(config)

    def generate(self, num_samples: int, training=False):
        """Generate new samples from random latent codes"""
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        generated = self.decoder(z, training=training)
        return generated

    def train_step(self, data):
        """
        Custom training step combining VAE and GAN losses
        """
        if isinstance(data, tuple):
            real_data = data[0]
        else:
            real_data = data

        # ========== Train Discriminator ==========
        with tf.GradientTape() as disc_tape:
            # Encode and reconstruct
            z_mean, z_log_var = self.encoder(real_data, training=True)
            z = self.reparameterize(z_mean, z_log_var)
            reconstructed = self.decoder(z, training=True)

            # Discriminator predictions
            real_logits = self.discriminator(real_data, training=True)
            fake_logits = self.discriminator(reconstructed, training=True)

            # Discriminator loss (binary cross-entropy)
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits), logits=real_logits
                )
            )
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_logits), logits=fake_logits
                )
            )
            disc_loss = real_loss + fake_loss

        # Update discriminator
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables))

        # ========== Train VAE (Encoder + Decoder) ==========
        with tf.GradientTape() as vae_tape:
            # Encode and reconstruct
            z_mean, z_log_var = self.encoder(real_data, training=True)
            z = self.reparameterize(z_mean, z_log_var)
            reconstructed = self.decoder(z, training=True)

            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.square(real_data - reconstructed)
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                              tf.exp(z_log_var), axis=1)
            )

            # GAN loss for generator (fool discriminator)
            fake_logits = self.discriminator(reconstructed, training=True)
            vae_gan_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(fake_logits), logits=fake_logits
                )
            )

            # Total VAE loss
            total_vae_loss = reconstruction_loss + kl_loss + 0.3 * vae_gan_loss

        # Update VAE (encoder + decoder)
        vae_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        vae_grads = vae_tape.gradient(total_vae_loss, vae_vars)
        self.vae_optimizer.apply_gradients(zip(vae_grads, vae_vars))

        # Update metrics
        self.total_loss_tracker.update_state(total_vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.vae_gan_loss_tracker.update_state(vae_gan_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Evaluation step"""
        real_data = data

        # Forward pass
        z_mean, z_log_var = self.encoder(real_data, training=False)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=False)

        # Losses
        reconstruction_loss = tf.reduce_mean(
            tf.square(real_data - reconstructed)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                          tf.exp(z_log_var), axis=1)
        )

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}


# ========== Helper function for data augmentation ==========
def augment_user_data(
    vae_gan: VAEGAN,
    user_samples: List[np.ndarray],
    num_synthetic: int = 10
) -> List[np.ndarray]:
    """
    Generate synthetic samples for a specific user using their real samples

    Args:
        vae_gan: Trained VAE-GAN model
        user_samples: List of real digraph sequences for the user
        num_synthetic: Number of synthetic samples to generate

    Returns:
        List of synthetic digraph sequences
    """
    # Encode user samples to get latent distribution
    user_batch = tf.constant(np.array(user_samples), dtype=tf.float32)
    z_mean, z_log_var = vae_gan.encoder(user_batch, training=False)

    # Sample from learned distribution
    synthetic_samples = []
    for _ in range(num_synthetic):
        # Sample latent code near user's distribution
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Decode to generate synthetic sample
        synthetic = vae_gan.decoder(z, training=False)
        synthetic_samples.append(synthetic.numpy())

    return synthetic_samples


if __name__ == "__main__":
    print("Testing VAE-GAN components...")

    # Test parameters
    BATCH_SIZE = 8
    SEQ_LEN = 50
    INPUT_DIM = 9
    LATENT_DIM = 64

    # Create dummy data
    dummy_data = tf.random.normal((BATCH_SIZE, SEQ_LEN, INPUT_DIM))

    # Test VAE-GAN
    print("\n=== Testing VAE-GAN ===")
    vae_gan = VAEGAN(input_dim=INPUT_DIM,
                     latent_dim=LATENT_DIM, seq_len=SEQ_LEN)
    vae_gan.compile(optimizer=keras.optimizers.Adam(5e-5))

    # Forward pass
    reconstructed, z_mean, z_log_var = vae_gan(dummy_data, training=False)
    print(f"Input shape: {dummy_data.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent mean shape: {z_mean.shape}")
    print(f"Latent log_var shape: {z_log_var.shape}")

    # Generate samples
    generated = vae_gan.generate(num_samples=5)
    print(f"Generated samples shape: {generated.shape}")

    # Test training step
    print("\n=== Testing Training Step ===")
    metrics = vae_gan.train_step(dummy_data)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n✓ VAE-GAN implementation complete!")
