"""
Visualization utilities for keystroke authentication embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

def visualize_embeddings_tsne(encoder, X_data, user_sessions, num_users=10, samples_per_user=50, 
                               title_prefix="", save_path=None):
    """
    Visualize the learned embedding space using t-SNE.
    
    Args:
        encoder: Trained encoder model
        X_data: List of input samples
        user_sessions: Dictionary mapping user_id to list of sample indices
        num_users: Number of users to visualize
        samples_per_user: Maximum samples per user
        title_prefix: Prefix for plot title (e.g., "Training" or "Test")
        save_path: Path to save the figure (optional)
    """
    # Select a subset of users
    user_ids = list(user_sessions.keys())[:num_users]
    
    embeddings_list = []
    labels = []
    colors_map = plt.cm.tab10(np.linspace(0, 1, num_users))
    
    print(f"Collecting embeddings for {num_users} users...")
    for idx, user_id in enumerate(user_ids):
        sessions = user_sessions[user_id][:samples_per_user]
        
        for session_idx in sessions:
            sample = tf.constant([X_data[session_idx]], dtype=tf.float32)
            embedding = encoder(sample, training=False).numpy()[0]
            embeddings_list.append(embedding)
            labels.append(idx)
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings_list)
    labels = np.array(labels)
    
    print(f"Computing t-SNE for {len(embeddings)} samples from {num_users} users...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    for idx, user_id in enumerate(user_ids):
        mask = labels == idx
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors_map[idx]], 
            label=f'User {user_id}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    
    title = f't-SNE Visualization of {title_prefix} Keystroke Embeddings'
    plt.title(title.strip(), fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    
    plt.show()
    
    return embeddings_2d, labels


def visualize_embeddings_umap(encoder, X_data, user_sessions, num_users=10, samples_per_user=50,
                               title_prefix="", save_path=None):
    """
    Visualize the learned embedding space using UMAP.
    Requires: pip install umap-learn
    
    Args:
        encoder: Trained encoder model
        X_data: List of input samples
        user_sessions: Dictionary mapping user_id to list of sample indices
        num_users: Number of users to visualize
        samples_per_user: Maximum samples per user
        title_prefix: Prefix for plot title (e.g., "Training" or "Test")
        save_path: Path to save the figure (optional)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("UMAP not installed. Run: pip install umap-learn")
    
    # Select a subset of users
    user_ids = list(user_sessions.keys())[:num_users]
    
    embeddings_list = []
    labels = []
    colors_map = plt.cm.tab10(np.linspace(0, 1, num_users))
    
    print(f"Collecting embeddings for {num_users} users...")
    for idx, user_id in enumerate(user_ids):
        sessions = user_sessions[user_id][:samples_per_user]
        
        for session_idx in sessions:
            sample = tf.constant([X_data[session_idx]], dtype=tf.float32)
            embedding = encoder(sample, training=False).numpy()[0]
            embeddings_list.append(embedding)
            labels.append(idx)
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings_list)
    labels = np.array(labels)
    
    print(f"Computing UMAP for {len(embeddings)} samples from {num_users} users...")
    
    # Apply UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, verbose=True)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    for idx, user_id in enumerate(user_ids):
        mask = labels == idx
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors_map[idx]], 
            label=f'User {user_id}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    
    title = f'UMAP Visualization of {title_prefix} Keystroke Embeddings'
    plt.title(title.strip(), fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    
    plt.show()
    
    return embeddings_2d, labels


def plot_training_history(history, save_path=None):
    """
    Plot training history including loss and accuracies.
    
    Args:
        history: Dictionary with keys 'train_losses', 'train_accs', 'val_accs'
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_losses'], label='Train Loss', alpha=0.7, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Episodes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_accs'], label='Train Accuracy', color='orange', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Accuracy over Episodes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy plot
    plt.subplot(1, 3, 3)
    episodes = np.arange(0, len(history['val_accs'])) * 100  # Assumes eval_interval = 100
    plt.plot(episodes, history['val_accs'], label='Validation Accuracy', 
             color='green', marker='o', markersize=3, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy over Episodes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_distance_distributions(genuine_distances, impostor_distances, save_path=None):
    """
    Plot histograms of genuine vs impostor distances.
    
    Args:
        genuine_distances: Array of distances for genuine pairs
        impostor_distances: Array of distances for impostor pairs
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    plt.hist(genuine_distances, bins=50, alpha=0.6, label='Genuine', color='green', edgecolor='black')
    plt.hist(impostor_distances, bins=50, alpha=0.6, label='Impostor', color='red', edgecolor='black')
    
    plt.axvline(np.mean(genuine_distances), color='green', linestyle='--', 
                linewidth=2, label=f'Genuine Mean: {np.mean(genuine_distances):.3f}')
    plt.axvline(np.mean(impostor_distances), color='red', linestyle='--', 
                linewidth=2, label=f'Impostor Mean: {np.mean(impostor_distances):.3f}')
    
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Genuine vs Impostor Distances', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance distribution plot saved to {save_path}")
    
    plt.show()