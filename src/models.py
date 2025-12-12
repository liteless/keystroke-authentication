import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List

class DigraphCNN(keras.Model):
    """
    1D CNN for embedding digraph sequences to capture temporal typing patterns.

    Input: variable-length sequences of digraphs 
    Output: fixed-size embeddings.
    """
    def __init__(
        self,
        input_dim: int = 9, # Number of digraph features
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 5, 7],
            # Kernel 3 → captures short-term timing patterns
            # Kernel 5 → captures mid-range digraph patterns
            # Kernel 7 → captures long-range rhythm
        num_filters: List[int] = [64, 128, 256],
        dropout: float = 0.3,
        **kwargs
    ):
        super(DigraphCNN, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.do_normalize = True
        
        # Multi-scale 1D convolutions
        self.conv_blocks = []
        for kernel_size in kernel_sizes:
            conv_block = keras.Sequential([
                layers.Conv1D(num_filters[0], kernel_size, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(dropout),
                
                layers.Conv1D(num_filters[1], kernel_size, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(dropout),
                
                layers.Conv1D(num_filters[2], kernel_size, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(dropout),
            ], name=f'conv_block_k{kernel_size}')
            self.conv_blocks.append(conv_block)
        
        # Global pooling to handle variable lengths of sequences
        self.global_pool = layers.GlobalMaxPooling1D()
        
        # Projection to embedding space
        self.fc = keras.Sequential([
            layers.Dense(embedding_dim * 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(embedding_dim)
        ], name='projection')
    
    def call(self, x, training=None):
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(x, training=training)
            pooled = self.global_pool(conv_out)
            conv_outputs.append(pooled)
        
        combined = tf.concat(conv_outputs, axis=1)
        embedding = self.fc(combined, training=training)
        
        # L2 normalize for metric learning
        if self.do_normalize:
            embedding = tf.nn.l2_normalize(embedding, axis=1)
        
        return embedding
    

def euclidean_distances(x, y):
    """Compute pairwise Euclidean distance between two sets of vectors"""
    x_expanded = tf.expand_dims(x, 1)  
    y_expanded = tf.expand_dims(y, 0)  

    distances_matrix = tf.sqrt(tf.reduce_sum(
        tf.square(x_expanded - y_expanded), axis=2) + 1e-8)

    return distances_matrix


if __name__ == "__main__":
    BATCH_SZ = 4
    SEQ_LEN = 50
    INPUT_DIM = 9

    dummy_data = tf.random.normal((BATCH_SZ, SEQ_LEN, INPUT_DIM))

    #------------------Test CNN encoder------------------
    print("Testing DigraphCNN...")
    model = DigraphCNN(
        input_dim=INPUT_DIM,
        embedding_dim=128,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256]
    )

    embeddings = model(dummy_data, training=False)
    print(f"Input shape: {dummy_data.shape}")
    print(f"Embedding shape: {embeddings.shape}")

    norms = tf.norm(embeddings, axis=1)
    print(f"Embeddings are normalized: {tf.reduce_all(tf.abs(norms - 1.0) < 0.01)}")

