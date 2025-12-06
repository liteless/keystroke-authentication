import os
import pickle
import tensorflow as tf
from tqdm import tqdm
from src.meta_learning import MetaLearningTrainer
from src.models import DigraphCNN
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer

# Make sure to change the below path to your local data path
DATA_ROOT = "data/UB_keystroke_dataset"
OUTPUT_PATH = "processed_keystrokes.pkl"


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

    trainer = MetaLearningTrainer(
        encoder=encoder,
        n_way=5,
        k_shot=1,
        q_query=1,
        lr=1e-3
    )

    print("\nStarting meta-learning training...")
    history = trainer.train(
        X_train_norm,
        y_train,
        user_sessions_train,
        num_episodes=1000,
        eval_interval=100,
        X_val=X_test_norm,
        y_val=y_test,
        user_sessions_val=user_sessions_test
    )

    # ---- STEP 4: FINAL EVALUATION ----
    print("\nFinal evaluation on test set...")
    test_acc = trainer.evaluate(
        X_test_norm,
        y_test,
        user_sessions_test,
        num_episodes=1000
    )
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
