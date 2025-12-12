import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src.models import DigraphCNN
import matplotlib.pyplot as plt

class VerificationMetaTrainer:
    """
    Few-shot verification trainer:
    - support: k_shot samples from target user X
    - query: 1 sample (same user X or different user Y)
    - label: 1 if same user, 0 otherwise
    """

    def __init__(self, encoder: DigraphCNN, k_shot: int, q_query: int, lr=1e-3):
        self.encoder = encoder
        self.k_shot = k_shot #number of support samples
        self.q_query = q_query #number of windows per query

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        #a verification head that transforms the Euclidean distance between two embeddings into a probability of being the same user
        self.verification_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),    
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1) #logit giving probability of being the same user
        ])

    def create_episode(self, digraphs, user_sessions, num_genuine=2, num_impostor=2):
        """
        Creates one verification episode with multiple windows per query sample.

        - pick a target user X
        - sample k_shot support windows from X
        - for each genuine query sample: sample `q_query` windows from X
        - for each impostor query sample: sample `q_query` windows from some Y != X

        Returns:
            support: (k_shot, T, D)
            query:   (num_samples * q_query, T, D), where num_samples = num_genuine + num_impostor
            labels:  (num_samples, 1)   (1 = genuine, 0 = impostor)
        """
        users = list(user_sessions.keys())
        if len(users) < 2:
            raise ValueError("Need at least 2 users for verification episodes.")

        valid_users = [u for u in users
                       if len(user_sessions[u]) >= self.k_shot + self.q_query]
        if not valid_users:
            raise ValueError("No users with enough sessions for this episode.")
        
        #step 1: choose target user X
        valid_users = [
            u for u in users
            if len(user_sessions[u]) >= self.k_shot + num_genuine * self.q_query
        ]
        if not valid_users:
            raise ValueError("No users with enough windows for this episode.")

        target_user = np.random.choice(valid_users)
        target_indices = user_sessions[target_user]


        #step 2: sample support + genuine query from target user X
        support_idx = np.random.choice(
            target_indices,
            self.k_shot,
            replace=False
        )
        genuine_groups = []
        remaining_for_genuine = list(
            set(target_indices) - set(support_idx)
        )
        for _ in range(num_genuine):
            if len(remaining_for_genuine) >= self.q_query:
                group = np.random.choice(
                    remaining_for_genuine, self.q_query, replace=False
                )
                # remove used ones, to reduce overlap between genuine samples
                remaining_for_genuine = list(
                    set(remaining_for_genuine) - set(group)
                )
            else:
                group = np.random.choice(
                    target_indices, self.q_query, replace=True
                )
            genuine_groups.append(group)


        #step 3: sample imposter indices from other users
        other_users = [u for u in users if u != target_user and len(user_sessions[u]) > 0]
        if not other_users:
            raise ValueError("No impostor users with sessions.")
        
        impostor_groups = []
        for _ in range(num_impostor):
            imp_user = np.random.choice(other_users)
            imp_indices = user_sessions[imp_user]
            if len(imp_indices) >= self.q_query:
                group = np.random.choice(imp_indices, self.q_query, replace=False)
            else:
                group = np.random.choice(imp_indices, self.q_query, replace=True)
            impostor_groups.append(group)

        #step 4: build samples
        support_samples = [digraphs[i] for i in support_idx]
        query_indices = []
        for group in genuine_groups:
            query_indices.extend(list(group))
        for group in impostor_groups:
            query_indices.extend(list(group))

        query_samples = [digraphs[i] for i in query_indices]

        #step 5: pad to uniform length
        support = self._pad_sequences(support_samples)
        query = self._pad_sequences(query_samples)

        #labels: genuine, then imposters
        labels = [1] * num_genuine + [0] * num_impostor
        labels = tf.constant(labels, dtype=tf.float32)
        labels = tf.reshape(labels, (-1, 1))

        return support, query, labels

    def _pad_sequences(self, sequences):
        max_len = max(seq.shape[0] for seq in sequences)
        padded = []

        for seq in sequences:
            if seq.shape[0] < max_len:
                padding = np.zeros(
                    (max_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded.append(padded_seq)

        return tf.constant(np.array(padded), dtype=tf.float32)
    
    def _forward_verification(self, support, query, training: bool):
        """
        support: (k_shot, seq_len, input_dim)
        query: (1, seq_len, input_dim)
        """
        #step 1: embed w/ CNN encoder
        support_embeddings = self.encoder(support, training=training) 
        query_embeddings = self.encoder(query, training=training)

        #step 2: prototype for user X = mean of support embeddings
        prototype = tf.reduce_mean(support_embeddings, axis=0, keepdims=True)

        total_q = tf.shape(query_embeddings)[0]
        emb_dim = tf.shape(query_embeddings)[1]
        q_windows = self.q_query

        num_samples = total_q // q_windows
        query_reshaped = tf.reshape(query_embeddings, (num_samples, q_windows, emb_dim))
        query_sample_emb = tf.reduce_mean(query_reshaped, axis=1)

        #step 3: compute Euclidean distance between query and prototype
        distances = tf.norm(query_sample_emb - prototype, axis=1, keepdims=True)

        #step 4: pass distance through verification head to get probability logits
        logits = self.verification_head(distances, training=training) 

        #step 5: return logits
        return logits, distances

    def train(self, X_train, user_sessions_train, num_episodes, eval_interval, X_val, user_sessions_val):
        train_losses = []
        train_accuracies = []
        val_accuracies = []

        window = 100 #episodes to avg over

        running_accuracy = 0.0
        alpha = 0.01 #smoothing factor for running average

        # Add tqdm progress bar
        pbar = tqdm(range(num_episodes), desc="Training Episodes", unit="episode")

        for episode in pbar:
            support, query, labels = self.create_episode(
                X_train, user_sessions_train, num_genuine=2, num_impostor=2)

            with tf.GradientTape() as tape:
                logits, _ = self._forward_verification(
                    support, query, training=True)
                loss = self.loss_fn(labels, logits)

            trainable_vars = (
                list(self.encoder.trainable_variables) +
                list(self.verification_head.trainable_variables)
            )
            grads = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))

            train_losses.append(loss.numpy())

            #obtain the accuracy for this episode
            probs = tf.sigmoid(logits)
            preds = tf.cast(probs >= 0.5, tf.float32)  
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(preds, labels), tf.float32))
            train_accuracies.append(accuracy.numpy())
           
            running_accuracy = alpha * accuracy.numpy() + (1 - alpha) * running_accuracy

            #rolling averages 
            start_idx = max(0, len(train_accuracies) - window)
            rolling_avg_acc = np.mean(train_accuracies[start_idx:])
            rolling_avg_loss = np.mean(train_losses[start_idx:])

            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss.numpy():.4f}',
                'train_acc': f'{accuracy.numpy():.4f}',
                'run_acc': f'{running_accuracy:.4f}',
                'rol_acc': f'{rolling_avg_acc:.4f}',
                'rol_loss': f'{rolling_avg_loss:.4f}'
            })
    
            if (episode + 1) % eval_interval == 0:
                val_accuracy = self.evaluate(
                    X_val, user_sessions_val, num_episodes=100)
                val_accuracies.append(val_accuracy)

                # Print validation results
                tqdm.write(
                    f"Episode {episode+1}/{num_episodes} - "
                    f"Loss: {loss.numpy():.4f}, "
                    f"Train Acc: {accuracy.numpy():.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )

                # Update progress bar with validation accuracy
                pbar.set_postfix({
                    'loss': f'{loss.numpy():.4f}',
                    'train_acc': f'{accuracy.numpy():.4f}',
                    'val_acc': f'{val_accuracy:.4f}'
                })

        pbar.close()

        return {
            'train_losses': train_losses,
            'train_accs': train_accuracies,
            'val_accs': val_accuracies
        }

    def evaluate(self, X_eval, user_sessions_eval, num_episodes=200):
        accuracy_list = []

        # Add tqdm progress bar for evaluation
        for _ in tqdm(range(num_episodes), desc="Evaluating", leave=False):
            try:
                support, query, labels = self.create_episode(
                    X_eval, user_sessions_eval, num_genuine=2, num_impostor=2
                )
            except ValueError:
                continue  #skip: not enough data

            logits, _ = self._forward_verification(
                support, query, training=False
            )

            probs = tf.sigmoid(logits)
            preds = tf.cast(probs >= 0.5, tf.float32)

            acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
            accuracy_list.append(acc.numpy())

        if not accuracy_list:
            return 0.0  #no valid episodes were run
        
        return tf.reduce_mean(accuracy_list)

    def authenticate(self, support_samples, query_samples, threshold):
        """
        support_samples: list of digraph arrays for user X (k_shot)
        query_samples:   list of digraph arrays (could be 1 or more)
        threshold:       probability threshold for "same user"

        Returns:
            authenticated: np.array of bools
            probs:         np.array of probabilities that sample is from X
        """

        support = self._pad_sequences(support_samples)
        query = self._pad_sequences(query_samples)

        support_embeddings = self.encoder(support, training=False)
        query_embeddings = self.encoder(query, training=False)

        prototype = tf.reduce_mean(support_embeddings, axis=0, keepdims=True)
        distances = tf.norm(query_embeddings - prototype, axis=1, keepdims=True)

        logits = self.verification_head(distances, training=False)
        probs = tf.sigmoid(logits) #how confident we are that query is from same user

        authenticated = probs.numpy().flatten() >= threshold

        return authenticated, probs.numpy().flatten()
    
    def _compute_confusion_counts(self, X_eval, user_sessions_eval, num_episodes: int, threshold: float):
        """
        Internal helper: run episodes and return raw confusion counts.
        """
        tp = tn = fp = fn = 0

        for _ in tqdm(range(num_episodes), desc="Evaluating", leave=False):
            try:
                support, query, labels = self.create_episode(
                    X_eval, user_sessions_eval, num_genuine=2, num_impostor=2
                )
            except ValueError:
                continue #skip: not enough data

            logits, _ = self._forward_verification(
                support, query, training=False
            )

            probs = tf.sigmoid(logits).numpy().flatten()
            preds = (probs >= threshold).astype(int)
            true_labels = labels.numpy().flatten().astype(int)

            for pred, true in zip(preds, true_labels):
                if true == 1 and pred == 1:
                    tp += 1
                elif true == 1 and pred == 0:
                    fn += 1
                elif true == 0 and pred == 0:
                    tn += 1
                elif true == 0 and pred == 1:
                    fp += 1

        return tp, tn, fp, fn
    
    def debug_probs(self, X_eval, user_sessions_eval, num_episodes=50, threshold=0.5):
        all_probs = []
        all_labels = []

        for _ in range(num_episodes):
            try:
                support, query, labels = self.create_episode(
                    X_eval, user_sessions_eval, num_genuine=2, num_impostor=2
                )
            except ValueError:
                continue

            logits, _ = self._forward_verification(
                support, query, training=False
            )
            probs = tf.sigmoid(logits).numpy().flatten()
            labs = labels.numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        print("Num samples:", len(all_probs))
        print("Label distribution: mean =", all_labels.mean(), " (should be ~0.5)")
        print("Prob stats: min =", all_probs.min(), " max =", all_probs.max(),
              " mean =", all_probs.mean())
        print("Fraction of probs >= 0.5:", np.mean(all_probs >= 0.5))
        print("Fraction of probs >= 0.7:", np.mean(all_probs >= 0.7))
        print("First 20 (label, prob):")
        for y, p in list(zip(all_labels, all_probs))[:20]:
            print(f"  y={int(y)}, p={p:.3f}")

    def debug_distances(self, X_eval, user_sessions_eval, num_episodes=100):
        genuine_d = []
        impostor_d = []

        for _ in range(num_episodes):
            try:
                support, query, labels = self.create_episode(
                    X_eval, user_sessions_eval,
                    num_genuine=2, num_impostor=2
                )
            except ValueError:
                continue

            _, distances = self._forward_verification(
                support, query, training=False
            )

            d = distances.numpy().flatten()
            labs = labels.numpy().flatten()

            for di, yi in zip(d, labs):
                if yi == 1:
                    genuine_d.append(di)
                else:
                    impostor_d.append(di)

        genuine_d = np.array(genuine_d)
        impostor_d = np.array(impostor_d)

        print("\n===== DEBUG DISTANCES =====")
        print("Num genuine:", len(genuine_d), " Num impostor:", len(impostor_d))
        if len(genuine_d) > 0:
            print("Genuine:  mean =", genuine_d.mean(), " std =", genuine_d.std())
        if len(impostor_d) > 0:
            print("Impostor: mean =", impostor_d.mean(), " std =", impostor_d.std())
        print("===========================\n")

    def evaluate_metrics(self,
                         X_eval,
                         user_sessions_eval,
                         num_episodes: int = 200,
                         threshold: float = 0.5):
        """
        Returns accuracy, TPR, TNR, FPR, FNR over many episodes.
        """
        tp, tn, fp, fn = self._compute_confusion_counts(
            X_eval, user_sessions_eval, num_episodes, threshold
        )

        total = tp + tn + fp + fn
        if total == 0:
            return {
                "accuracy": 0.0,
                "tpr": 0.0,
                "tnr": 0.0,
                "fpr": 0.0,
                "fnr": 0.0,
            }

        accuracy = (tp + tn) / total
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 

        return {
            "accuracy": float(accuracy),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "fpr": float(fpr),
            "fnr": float(fnr),
        }
    
    def compute_confusion(self, X_eval, user_sessions_eval, num_episodes: int = 500, threshold: float = 0.5):
        """
        Returns confusion matrix:
            [[tn, fp],
             [fn, tp]]
        """
        tp, tn, fp, fn = self._compute_confusion_counts(
            X_eval, user_sessions_eval, num_episodes, threshold
        )
        return np.array([[tn, fp], [fn, tp]])
    
    def plot_confusion_matrix(self, cm):
        """
        cm = confusion matrix:
            [[tn, fp],
             [fn, tp]]
        """
        _, ax = plt.subplots(figsize=(5, 5))

        # Show values inside the matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Impostor", "Pred: Genuine"])
        ax.set_yticklabels(["True: Impostor", "True: Genuine"])

        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title("Verification Confusion Matrix", fontsize=16)

        plt.tight_layout()
        plt.show()