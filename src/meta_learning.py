import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src.models import DigraphCNN, PrototypicalNetwork


class MetaLearningTrainer:
    def __init__(self, encoder: DigraphCNN, n_way: int, k_shot: int, q_query: int, lr=1e-3):
        self.encoder = encoder
        self.prototypical_network = PrototypicalNetwork(encoder)

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    def create_episode(self, digraphs, user_ids, user_sessions):
        active_users = list(user_sessions.keys())

        valid_users = [u for u in active_users
                       if len(user_sessions[u]) >= self.k_shot + self.q_query]

        if len(valid_users) < self.n_way:
            raise ValueError(
                f"Not enough users with {self.k_shot + self.q_query} sessions. "
                f"Found {len(valid_users)} valid users, need {self.n_way}"
            )

        selected_users = np.random.choice(
            valid_users, self.n_way, replace=False)

        support_samples = []
        support_labels = []
        query_samples = []
        query_labels = []

        for i, user_id in enumerate(selected_users):
            session_indices = user_sessions[user_id]

            sampled_idx = np.random.choice(
                session_indices,
                self.k_shot + self.q_query,
                replace=False
            )

            support_idx = sampled_idx[:self.k_shot]
            query_idx = sampled_idx[self.k_shot:]

            for idx in support_idx:
                support_samples.append(digraphs[idx])
                support_labels.append(i)

            for idx in query_idx:
                query_samples.append(digraphs[idx])
                query_labels.append(i)

        support_samples = self._pad_sequences(support_samples)
        query_samples = self._pad_sequences(query_samples)

        support_labels = tf.constant(support_labels, dtype=tf.int32)
        query_labels = tf.constant(query_labels, dtype=tf.int32)

        return support_samples, support_labels, query_samples, query_labels

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

    def train(self, X_train, y_train, user_sessions_train, num_episodes, eval_interval, X_val, y_val, user_sessions_val):
        train_losses = []
        train_accuracies = []
        val_accuracies = []

        # Add tqdm progress bar
        pbar = tqdm(range(num_episodes), desc="Training Episodes", unit="episode")

        for episode in pbar:
            support, support_labels, query, query_labels = self.create_episode(
                X_train, y_train, user_sessions_train)

            with tf.GradientTape() as tape:
                logits, _ = self.prototypical_network(
                    (support, support_labels, query), training=True)
                loss = self.loss_fn(query_labels, logits)

            grads = tape.gradient(loss,
                                  self.prototypical_network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.prototypical_network.trainable_variables))

            train_losses.append(loss.numpy())

            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(preds, query_labels), tf.float32))
            train_accuracies.append(accuracy.numpy())

            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss.numpy():.4f}',
                'train_acc': f'{accuracy.numpy():.4f}'
            })

            if (episode + 1) % eval_interval == 0:
                val_accuracy = self.evaluate(
                    X_val, y_val, user_sessions_val)
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

    def evaluate(self, X_eval, y_eval, user_sessions_eval, num_episodes=100):
        accuracy_list = []

        # Add tqdm progress bar for evaluation
        for _ in tqdm(range(num_episodes), desc="Evaluating", leave=False):
            support, support_labels, query, query_labels = self.create_episode(
                X_eval, y_eval, user_sessions_eval)

            logits, _ = self.prototypical_network(
                (support, support_labels, query), training=False)

            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(preds, query_labels), tf.float32))
            accuracy_list.append(accuracy.numpy())

        return tf.reduce_mean(accuracy_list)

    def authenticate(self, support_samples, query_samples, threshold):

        support = self._pad_sequences(support_samples)
        query = self._pad_sequences(query_samples)

        support_embeddings = self.encoder(support, training=False)
        query_embeddings = self.encoder(query, training=False)

        prototype = tf.reduce_mean(support_embeddings, axis=0, keepdims=True)

        distances = tf.norm(query_embeddings - prototype, axis=1)

        authenticated = distances < threshold
        confidence = 1.0 - (distances / threshold)

        return authenticated.numpy(), confidence.numpy()
