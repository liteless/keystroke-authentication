"""
Standalone evaluation script for trained VAE-GAN models.

Usage:
    python src/evaluation.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gan import VAEGAN
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer


def pad_sequences_to_max(sequences, max_len):
    """Pad variable-length sequences to max_len"""
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            padding = np.zeros((max_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    return np.array(padded, dtype=np.float32)


def evaluate_reconstruction_quality(vae_gan, X_test, num_samples=100, max_seq_len=80):
    """
    Evaluate how well the VAE reconstructs real samples.
    
    Returns:
        dict: Reconstruction metrics (MSE, MAE, feature-wise errors)
    """
    print("\n=== Evaluating Reconstruction Quality ===")
    
    # Sample random test data
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    test_samples_list = [X_test[i] for i in indices]
    
    # Pad to max length
    test_samples = pad_sequences_to_max(test_samples_list, max_seq_len)
    test_batch = tf.constant(test_samples, dtype=tf.float32)
    
    # Reconstruct
    reconstructed, _, _ = vae_gan(test_batch, training=False)
    reconstructed = reconstructed.numpy()
    
    # Compute metrics
    mse = mean_squared_error(test_samples.reshape(-1), reconstructed.reshape(-1))
    mae = mean_absolute_error(test_samples.reshape(-1), reconstructed.reshape(-1))
    
    # Per-feature MSE (9 features)
    feature_mse = []
    feature_names = ['Horiz Dist', 'Vert Dist', 'Euclid Dist', 
                     'Hold1', 'Hold2', 'Latency', 'PP', 'RR', 'PR']
    for i in range(9):
        feat_mse = mean_squared_error(test_samples[:, :, i].reshape(-1), 
                                       reconstructed[:, :, i].reshape(-1))
        feature_mse.append(feat_mse)
    
    results = {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'feature_mse': dict(zip(feature_names, feature_mse))
    }
    
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"MAE:  {mae:.6f}")
    print("\nPer-Feature MSE:")
    for name, val in results['feature_mse'].items():
        print(f"  {name:15s}: {val:.6f}")
    
    return results


def evaluate_generation_quality(vae_gan, X_real, num_synthetic=100, max_seq_len=80):
    """
    Evaluate quality of generated samples using distributional similarity.
    
    Returns:
        dict: Generation metrics (Wasserstein distances, statistical moments)
    """
    print("\n=== Evaluating Generation Quality ===")
    
    # Generate synthetic samples
    synthetic = vae_gan.generate(num_synthetic, training=False).numpy()
    
    # Sample equal number of real samples
    indices = np.random.choice(len(X_real), num_synthetic, replace=False)
    real_list = [X_real[i] for i in indices]
    real = pad_sequences_to_max(real_list, max_seq_len)
    
    # Compute Wasserstein distance for each feature
    feature_names = ['Horiz Dist', 'Vert Dist', 'Euclid Dist', 
                     'Hold1', 'Hold2', 'Latency', 'PP', 'RR', 'PR']
    
    wasserstein_dists = []
    for i in range(9):
        real_feat = real[:, :, i].reshape(-1)
        synth_feat = synthetic[:, :, i].reshape(-1)
        wd = wasserstein_distance(real_feat, synth_feat)
        wasserstein_dists.append(wd)
    
    # Statistical moments comparison
    real_mean = real.mean(axis=(0, 1))
    synth_mean = synthetic.mean(axis=(0, 1))
    real_std = real.std(axis=(0, 1))
    synth_std = synthetic.std(axis=(0, 1))
    
    results = {
        'wasserstein_distances': dict(zip(feature_names, wasserstein_dists)),
        'mean_wasserstein': np.mean(wasserstein_dists),
        'real_mean': real_mean,
        'synth_mean': synth_mean,
        'real_std': real_std,
        'synth_std': synth_std
    }
    
    print(f"Mean Wasserstein Distance: {np.mean(wasserstein_dists):.6f}")
    print("\nPer-Feature Wasserstein Distance:")
    for name, val in results['wasserstein_distances'].items():
        print(f"  {name:15s}: {val:.6f}")
    
    return results


def evaluate_latent_space(vae_gan, X_data, user_sessions, num_users=10, max_seq_len=80):
    """
    Evaluate latent space structure - do same-user samples cluster?
    
    Returns:
        dict: Intra-user vs inter-user distances
    """
    print("\n=== Evaluating Latent Space Structure ===")
    
    user_ids = list(user_sessions.keys())[:num_users]
    
    # Encode samples from each user
    user_embeddings = {}
    for user_id in user_ids:
        samples_list = [X_data[i] for i in user_sessions[user_id][:20]]  # 20 samples per user
        samples = pad_sequences_to_max(samples_list, max_seq_len)
        batch = tf.constant(samples, dtype=tf.float32)
        z_mean, _ = vae_gan.encoder(batch, training=False)
        user_embeddings[user_id] = z_mean.numpy()
    
    # Compute intra-user distances (within same user)
    intra_dists = []
    for user_id, embeddings in user_embeddings.items():
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = euclidean(embeddings[i], embeddings[j])
                intra_dists.append(dist)
    
    # Compute inter-user distances (between different users)
    inter_dists = []
    user_list = list(user_embeddings.keys())
    for i in range(len(user_list)):
        for j in range(i+1, len(user_list)):
            emb1 = user_embeddings[user_list[i]]
            emb2 = user_embeddings[user_list[j]]
            # Compare centroids
            centroid1 = emb1.mean(axis=0)
            centroid2 = emb2.mean(axis=0)
            dist = euclidean(centroid1, centroid2)
            inter_dists.append(dist)
    
    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)
    separation_ratio = inter_mean / intra_mean  # Higher is better
    
    results = {
        'intra_user_mean': intra_mean,
        'intra_user_std': np.std(intra_dists),
        'inter_user_mean': inter_mean,
        'inter_user_std': np.std(inter_dists),
        'separation_ratio': separation_ratio
    }
    
    print(f"Intra-user distance: {intra_mean:.4f} ± {np.std(intra_dists):.4f}")
    print(f"Inter-user distance: {inter_mean:.4f} ± {np.std(inter_dists):.4f}")
    print(f"Separation ratio: {separation_ratio:.4f} (higher is better)")
    
    return results


def evaluate_discriminator_accuracy(vae_gan, X_real, num_samples=100, max_seq_len=80):
    """
    Evaluate discriminator's ability to distinguish real vs fake.
    Ideally should be ~50% if generator is good.
    """
    print("\n=== Evaluating Discriminator Performance ===")
    
    # Get real samples
    indices = np.random.choice(len(X_real), num_samples, replace=False)
    real_list = [X_real[i] for i in indices]
    real_padded = pad_sequences_to_max(real_list, max_seq_len)
    real_batch = tf.constant(real_padded, dtype=tf.float32)
    
    # Generate fake samples
    fake_batch = vae_gan.generate(num_samples, training=False)
    
    # Get discriminator predictions
    real_logits = vae_gan.discriminator(real_batch, training=False).numpy()
    fake_logits = vae_gan.discriminator(fake_batch, training=False).numpy()
    
    # Convert to probabilities
    real_probs = 1 / (1 + np.exp(-real_logits))  # Sigmoid
    fake_probs = 1 / (1 + np.exp(-fake_logits))
    
    # Accuracy (real should be >0.5, fake should be <0.5)
    real_acc = (real_probs > 0.5).mean()
    fake_acc = (fake_probs < 0.5).mean()
    overall_acc = (real_acc + fake_acc) / 2
    
    results = {
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'overall_accuracy': overall_acc,
        'real_prob_mean': real_probs.mean(),
        'fake_prob_mean': fake_probs.mean()
    }
    
    print(f"Real accuracy: {real_acc:.4f} (classify real as real)")
    print(f"Fake accuracy: {fake_acc:.4f} (classify fake as fake)")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Real prob mean: {real_probs.mean():.4f}")
    print(f"Fake prob mean: {fake_probs.mean():.4f}")
    print("\nNote: ~50% discriminator accuracy suggests well-trained generator")
    
    return results


def visualize_vae_reconstructions(vae_gan, X_data, num_samples=5, max_seq_len=80, save_path=None):
    """
    Visualize original vs reconstructed samples.
    """
    print("\n=== Generating Reconstruction Visualizations ===")
    
    indices = np.random.choice(len(X_data), num_samples)
    samples_list = [X_data[i] for i in indices]
    samples = pad_sequences_to_max(samples_list, max_seq_len)
    batch = tf.constant(samples, dtype=tf.float32)
    
    reconstructed, _, _ = vae_gan(batch, training=False)
    reconstructed = reconstructed.numpy()
    
    _, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(samples[i].T, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'Original Sample {i+1}', fontweight='bold')
        axes[i, 0].set_ylabel('Features')
        axes[i, 0].set_xlabel('Time')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed[i].T, aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed Sample {i+1}', fontweight='bold')
        axes[i, 1].set_xlabel('Time')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def plot_feature_distributions(vae_gan, X_real, num_synthetic=500, max_seq_len=80, save_path=None):
    """
    Compare feature distributions: real vs synthetic.
    """
    print("\n=== Generating Feature Distribution Plots ===")
    
    # Generate synthetic
    synthetic = vae_gan.generate(num_synthetic, training=False).numpy()
    
    # Sample real
    indices = np.random.choice(len(X_real), num_synthetic, replace=False)
    real_list = [X_real[i] for i in indices]
    real = pad_sequences_to_max(real_list, max_seq_len)
    
    feature_names = ['Horiz Dist', 'Vert Dist', 'Euclid Dist', 
                     'Hold1', 'Hold2', 'Latency', 'PP', 'RR', 'PR']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        real_feat = real[:, :, i].reshape(-1)
        synth_feat = synthetic[:, :, i].reshape(-1)
        
        axes[i].hist(real_feat, bins=50, alpha=0.5, label='Real', color='blue', density=True)
        axes[i].hist(synth_feat, bins=50, alpha=0.5, label='Synthetic', color='red', density=True)
        axes[i].set_title(name, fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Real vs Synthetic Feature Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def main():
    """
    Main evaluation pipeline
    """
    print("=" * 70)
    print("VAE-GAN EVALUATION SCRIPT")
    print("=" * 70)
    
    # Configuration
    DATA_ROOT = "data/UB_keystroke_dataset"
    WINDOW_SIZE = 80  # Must match training
    LATENT_DIM = 128  # ← ADD THIS - must match your trained model
    MODEL_DIR = "models"
    OUTPUT_DIR = "models/evaluation"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading and preprocessing data...")
    data = build_datasets(DATA_ROOT, window_size=WINDOW_SIZE)
    X_train, _, X_test, _, user_sessions_train, _ = data
    
    # Normalize
    print("[2/6] Normalizing data...")
    stats = fit_normalizer(X_train)
    X_train_norm = apply_normalizer(X_train, stats)
    X_test_norm = apply_normalizer(X_test, stats)
    
    print(f"Training samples: {len(X_train_norm)}")
    print(f"Test samples: {len(X_test_norm)}")
    print(f"Sequence length: {X_train_norm[0].shape[0]}")
    
    # Load VAE-GAN
    print("\n[3/6] Loading trained VAE-GAN model...")
    max_seq_len = max(x.shape[0] for x in X_train_norm)
    vae_gan = VAEGAN(input_dim=9, latent_dim=LATENT_DIM, seq_len=max_seq_len)  # ← CHANGED
    
    # Build model components properly
    dummy_input = tf.zeros((1, max_seq_len, 9), dtype=tf.float32)
    
    # Build encoder
    _ = vae_gan.encoder(dummy_input, training=False)
    
    # Build decoder
    dummy_z = tf.zeros((1, LATENT_DIM), dtype=tf.float32)  # ← CHANGED (was 64)
    _ = vae_gan.decoder(dummy_z, training=False)
    
    # Build discriminator
    _ = vae_gan.discriminator(dummy_input, training=False)
    
    print(f"Model built with seq_len={max_seq_len}, latent_dim={LATENT_DIM}")  # ← CHANGED
    
    # Load weights
    try:
        vae_gan.encoder.load_weights(f'{MODEL_DIR}/vae_encoder_weights.h5')
        vae_gan.decoder.load_weights(f'{MODEL_DIR}/vae_decoder_weights.h5')
        vae_gan.discriminator.load_weights(f'{MODEL_DIR}/vae_discriminator_weights.h5')
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        print("\nTroubleshooting:")
        print(f"  - Check that weights exist in '{MODEL_DIR}/'")
        print(f"  - Ensure VAE-GAN was trained with window_size={WINDOW_SIZE}")
        print(f"  - Current model seq_len: {max_seq_len}")
        return
    
    # Run evaluations
    print("\n" + "=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)
    
    print("\n[4/6] Quantitative Metrics...")
    
    # Reconstruction quality
    recon_metrics = evaluate_reconstruction_quality(vae_gan, X_test_norm, num_samples=200, max_seq_len=max_seq_len)
    
    # Generation quality
    gen_metrics = evaluate_generation_quality(vae_gan, X_train_norm, num_synthetic=300, max_seq_len=max_seq_len)
    
    # Latent space structure
    latent_metrics = evaluate_latent_space(vae_gan, X_train_norm, user_sessions_train, num_users=20, max_seq_len=max_seq_len)
    
    # Discriminator performance
    disc_metrics = evaluate_discriminator_accuracy(vae_gan, X_test_norm, num_samples=200, max_seq_len=max_seq_len)
    
    # Visual evaluations
    print("\n[5/6] Generating visualizations...")
    
    visualize_vae_reconstructions(vae_gan, X_test_norm, num_samples=5, max_seq_len=max_seq_len,
                                  save_path=f'{OUTPUT_DIR}/vae_reconstructions.png')
    
    plot_feature_distributions(vae_gan, X_train_norm, num_synthetic=500, max_seq_len=max_seq_len,
                               save_path=f'{OUTPUT_DIR}/feature_distributions.png')
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n✓ Reconstruction Quality:")
    print(f"  RMSE: {recon_metrics['rmse']:.6f}")
    
    print("\n✓ Generation Quality:")
    print(f"  Mean Wasserstein Distance: {gen_metrics['mean_wasserstein']:.6f}")
    
    print("\n✓ Latent Space:")
    print(f"  Separation Ratio: {latent_metrics['separation_ratio']:.4f}")
    
    print("\n✓ Discriminator:")
    print(f"  Overall Accuracy: {disc_metrics['overall_accuracy']:.4f}")
    
    print("\n✓ Visualizations saved to:", OUTPUT_DIR)
    
    print("\n[6/6] Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()