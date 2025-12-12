from src.normalization import fit_normalizer, apply_normalizer
from src.preprocessing import build_datasets
from src.gan import VAEGAN, augment_user_data
import numpy as np
import tensorflow as tf


def create_tf_dataset(X, batch_size=32, shuffle=True):
    """Create TensorFlow dataset from list of variable-length sequences"""
    # Pad all sequences to same length
    max_len = max(x.shape[0] for x in X)

    X_padded = []
    for x in X:
        if x.shape[0] < max_len:
            padding = np.zeros(
                (max_len - x.shape[0], x.shape[1]), dtype=np.float32)
            x_padded = np.vstack([x, padding])
        else:
            x_padded = x
        X_padded.append(x_padded)

    X_padded = np.array(X_padded, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(X_padded)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def pad_sequences(sequences, max_len=None):
    """Pad sequences to max_len"""
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)

    padded = []
    for seq in sequences:
        # Truncate or pad length
        if seq.shape[0] < max_len:
            padding = np.zeros((max_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)

    return np.array(padded), max_len


def main():
    # Load and preprocess data
    print("Loading data...")
    DATA_ROOT = "data/UB_keystroke_dataset"
    data = build_datasets(DATA_ROOT, window_size=80)  # ADD window_size=80
    X_train, _, X_test, _, user_sessions_train, _ = data

    # Normalize
    print("Normalizing data...")
    stats = fit_normalizer(X_train)
    X_train_norm = apply_normalizer(X_train, stats)
    X_test_norm = apply_normalizer(X_test, stats)

    # Get sequence length (use max from training set)
    max_seq_len = max(x.shape[0] for x in X_train_norm)
    print(f"Maximum sequence length: {max_seq_len}")

    # Pad sequences
    print("Padding sequences...")
    X_train_padded, max_len = pad_sequences(X_train_norm)
    X_test_padded, _ = pad_sequences(X_test_norm, max_len=max_len)

    # Create datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_tf_dataset(
        X_train_padded, batch_size=32, shuffle=True)
    val_dataset = create_tf_dataset(X_test_padded, batch_size=32, shuffle=False)

    # Create VAE-GAN model
    print("\nCreating VAE-GAN model...")
    vae_gan = VAEGAN(
        input_dim=9,
        latent_dim=128,
        seq_len=max_len
    )

    vae_gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # Train
    print("\nTraining VAE-GAN...")
    vae_gan.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        verbose=1
    )

    # Save model
    print("\nSaving model...")
    vae_gan.encoder.save_weights('vae_encoder_weights.h5')
    vae_gan.decoder.save_weights('vae_decoder_weights.h5')
    vae_gan.discriminator.save_weights('vae_discriminator_weights.h5')

    # Test generation
    print("\n=== Testing Synthetic Generation ===")
    user_id = list(user_sessions_train.keys())[0]
    user_session_indices = user_sessions_train[user_id][:5]
    user_samples = [X_train_norm[i] for i in user_session_indices]

    # Pad user samples to match training length
    user_samples_padded, _ = pad_sequences(user_samples, max_len=max_len)
    user_samples_list = [user_samples_padded[i] for i in range(len(user_samples_padded))]

    print(f"Generating synthetic samples for User {user_id}...")
    synthetic_samples = augment_user_data(
        vae_gan, user_samples_list, num_synthetic=10)
    print(f"Generated {len(synthetic_samples)} synthetic samples")

    print("\n✓ VAE-GAN training complete!")


if __name__ == "__main__":
    main()
