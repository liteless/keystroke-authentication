import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src.meta_learning import MetaLearningTrainer
from src.meta_learning2 import VerificationMetaTrainer
from src.models import DigraphCNN
from src.gan import VAEGAN
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer
import matplotlib.pyplot as plt

# Make sure to change the below path to your local data path
DATA_ROOT = "data/UB_keystroke_dataset"
OUTPUT_PATH = "processed_keystrokes.pkl"


def augment_user_data(vae_gan, user_samples, num_synthetic):
    """Generate synthetic samples for a user using the trained VAE-GAN."""
    # Stack user samples into a batch
    user_batch = np.array(user_samples)
    user_batch = tf.constant(user_batch, dtype=tf.float32)
    
    # Encode to get latent distribution
    z_mean, z_log_var = vae_gan.encoder(user_batch, training=False)
    
    # Generate synthetic samples
    synthetic_samples = []
    for _ in range(num_synthetic):
        # Sample from the learned distribution
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Decode to generate synthetic sample
        synthetic = vae_gan.decoder(z, training=False)
        synthetic_samples.append(synthetic.numpy())
    
    return synthetic_samples


def main():
    # ---- STEP 1: PREPROCESS DATA, OR LOAD PREPROCESSED DATA ----
    # if os.path.exists(OUTPUT_PATH):
    #     with open(OUTPUT_PATH, "rb") as f:
    #         data = pickle.load(f)

    #     print("Loaded preprocessed data from", OUTPUT_PATH)

    # else: #Preprocess from scratch
    #     data = build_datasets(DATA_ROOT)
    #     with open(OUTPUT_PATH, "wb") as f:
    #         pickle.dump(data, f)

    #     print("Created preprocessed data and saved to", OUTPUT_PATH)
    data = build_datasets(DATA_ROOT)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    user_sessions_train = data[4]
    user_sessions_test = data[5]

    # Confirm data shapes
    print("Training data length:", len(X_train))
    print("Training labels length:", len(y_train))
    print("Testing data length:", len(X_test))
    print("Testing labels length:", len(y_test))
    print("Number of users in training set:", len(user_sessions_train))
    print("Number of users in testing set:", len(user_sessions_test))
    print("Example digraph shape (train[0]):", X_train[0].shape)

    # ---- STEP 2: NORMALIZE DATA VIA LOG TRANSFORM & MEAN/STD SCALING ----
    # Get normalization stats from training data only
    stats = fit_normalizer(X_train)

    # Apply normalization to both train and test data
    X_train_norm = apply_normalizer(X_train, stats)
    X_test_norm = apply_normalizer(X_test, stats)

    # ---- STEP 3: CREATE ENCODER AND META-LEARNING TRAINER ----
    encoder = DigraphCNN(
        input_dim=9,
        embedding_dim=128,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256],
        dropout=0.3
    )

    # ---- STEP 4: TRAIN THE META-LEARNING MODEL ----
    trainer = VerificationMetaTrainer(
        encoder=encoder,
        k_shot=2,
        q_query=15,
        lr=1e-3
    )

    # ---- AUGMENT DATA WITH VAE-GAN ----
    print("\n---- Augmenting data with VAE-GAN ----")

    # Get max sequence length from training data
    max_seq_len = max(x.shape[0] for x in X_train_norm)

    # Create VAE-GAN model
    vae_gan = VAEGAN(input_dim=9, latent_dim=64, seq_len=max_seq_len)

    # Build the model by calling it with dummy data (REQUIRED before loading weights)
    dummy_input = tf.zeros((1, max_seq_len, 9), dtype=tf.float32)
    _ = vae_gan(dummy_input, training=False)  # This builds the model

    # Now load the weights
    vae_gan.encoder.load_weights('models/vae_encoder_weights.h5')
    vae_gan.decoder.load_weights('models/vae_decoder_weights.h5')

    # Find users with limited samples
    MIN_SAMPLES = 10
    for user_id, sessions in user_sessions_train.items():
        if len(sessions) < MIN_SAMPLES:
            user_samples = [X_train_norm[i] for i in sessions]
            synthetic = augment_user_data(vae_gan, user_samples, num_synthetic=MIN_SAMPLES-len(sessions))
            
            # Add synthetic samples to dataset
            for synth_sample in synthetic:
                X_train_norm.append(synth_sample[0])
                y_train.append(user_id)
                user_sessions_train[user_id].append(len(X_train_norm)-1)

    print("\nStarting meta-learning training...")
    history = trainer.train(
        X_train_norm,
        user_sessions_train,
        num_episodes=1000,
        eval_interval=100,
        X_test_norm,
        user_sessions_test
    )

    print("\n>>> DEBUGGING ON VALIDATION SET <<<")
    trainer.debug_probs(X_test_norm, user_sessions_test, num_episodes=100, threshold=0.5)

    print("\n>>> DEBUGGING ON TRAINING SET <<<")
    trainer.debug_probs(X_train_norm, user_sessions_train, num_episodes=100, threshold=0.5)

    print("\n>>> DISTANCES ON VALIDATION SET <<<")
    trainer.debug_distances(X_test_norm, user_sessions_test, num_episodes=100)

    # ---- STEP 5: VERIFICATION ----
    metrics = trainer.evaluate_metrics(
        X_test_norm,
        user_sessions_test,
        num_episodes=500,
        threshold=0.5,
    )
    
    print("Verification metrics on test set:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  TPR: {metrics['tpr']:.4f}")
    print(f"  TNR: {metrics['tnr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  FNR: {metrics['fnr']:.4f}")

    # ---- STEP 6: PLOT CONFUSION MATRIX & TRAINING HISTORY ----
    cm = trainer.compute_confusion(
        X_test_norm,
        user_sessions_test,
        num_episodes=500,
        threshold=0.5,
    )

    print("Confusion matrix:\n", cm)
    trainer.plot_confusion_matrix(cm, class_names=["Impostor", "Genuine"])

    #Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
