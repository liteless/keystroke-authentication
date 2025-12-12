import numpy as np
import tensorflow as tf
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer
from src.gan import VAEGAN

def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            padding = np.zeros((max_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    return np.array(padded), max_len


def evaluate_gan_accuracy(vae_gan, X_val, batch_size=32):
    reconstruction_losses = []
    kl_losses = []
    disc_real_accs = []
    disc_fake_accs = []
    
    num_batches = (len(X_val) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_val))
        batch = tf.constant(X_val[start_idx:end_idx], dtype=tf.float32)
        
        z_mean, z_log_var = vae_gan.encoder(batch, training=False)
        z = vae_gan.reparameterize(z_mean, z_log_var)
        reconstructed = vae_gan.decoder(z, training=False)
        
        real_logits = vae_gan.discriminator(batch, training=False)
        fake_logits = vae_gan.discriminator(reconstructed, training=False)
        
        recon_loss = tf.reduce_mean(tf.square(batch - reconstructed), axis=[1, 2])
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        
        real_preds = tf.cast(real_logits > 0, tf.float32)
        fake_preds = tf.cast(fake_logits <= 0, tf.float32)
        
        reconstruction_losses.extend(recon_loss.numpy())
        kl_losses.extend(kl_loss.numpy())
        disc_real_accs.append(tf.reduce_mean(real_preds).numpy())
        disc_fake_accs.append(tf.reduce_mean(fake_preds).numpy())
    
    return {
        'reconstruction_loss': np.mean(reconstruction_losses),
        'kl_loss': np.mean(kl_losses),
        'discriminator_real_accuracy': np.mean(disc_real_accs),
        'discriminator_fake_accuracy': np.mean(disc_fake_accs),
        'discriminator_overall_accuracy': (np.mean(disc_real_accs) + np.mean(disc_fake_accs)) / 2
    }


DATA_ROOT = "data/UB_keystroke_dataset"
data = build_datasets(DATA_ROOT, window_size=80)
X_train, _, X_test, _, _, _ = data

stats = fit_normalizer(X_train)
X_test_norm = apply_normalizer(X_test, stats)

# Use the same sequence length as during training
max_len = 50  # Standard sequence length from training
X_test_padded, _ = pad_sequences(X_test_norm, max_len=max_len)

vae_gan = VAEGAN(input_dim=9, latent_dim=64, seq_len=max_len)
dummy_input = tf.zeros((1, max_len, 9))
_ = vae_gan(dummy_input, training=False)
_ = vae_gan.discriminator(dummy_input, training=False)

vae_gan.encoder.load_weights('models/vae_encoder_weights.h5')
vae_gan.decoder.load_weights('models/vae_decoder_weights.h5')
vae_gan.discriminator.load_weights('models/vae_discriminator_weights.h5')

metrics = evaluate_gan_accuracy(vae_gan, X_test_padded)

print(f"Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
print(f"KL Loss: {metrics['kl_loss']:.6f}")
print(f"Discriminator Real Accuracy: {metrics['discriminator_real_accuracy']:.4f}")
print(f"Discriminator Fake Accuracy: {metrics['discriminator_fake_accuracy']:.4f}")
print(f"Discriminator Overall Accuracy: {metrics['discriminator_overall_accuracy']:.4f}")
