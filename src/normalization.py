import numpy as np
from typing import List, Tuple, Dict

#Digraphs hold the following 9 features: 
# 0. Horizontal distance between key2 and key1
# 1. Vertical distance between key2 and key1
# 2. Euclidean distance between key2 and key1
# 3. Key1 hold time (release - press)
# 4. Key2 hold time (release - press)
# 5. Inner-key time (key2_press - key1_release)
# 6. Outer-key time (key2_release - key1_press)
# 7. Keydown-to-keydown time (key2_press - key1_press)
# 8. Keyup-to-keyup time (key2_release - key1_release)

TIME_FEATURES = [3, 4, 5, 6, 7, 8]

def fit_normalizer(X_train: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute per-feature mean and standard deviation from the training set.
    Applies log1p to only time features.

    Args: 
        X_train: List of training samples, each sample is a numpy array of shape (num_digraphs, 9)

    Returns: 
        stats: Dictionary with keys 'mean' and 'std', each mapping to a numpy array of shape (9,)
      across the training data.
    """
    all_train = np.vstack(X_train)  # (total_digraphs, 9)
    all_train_copy = all_train.copy()

    #Apply ln(1 + x) to only time features for normalization 
    for idx in TIME_FEATURES:
        all_train_copy[:, idx] = np.log1p(np.abs(all_train_copy[:, idx])) * np.sign(all_train_copy[:, idx]) #ensure we preserve the sign after normalization

    #Compute mean and std for all features
    mean = np.mean(all_train_copy, axis=0)  # (9,)
    std = np.std(all_train_copy, axis=0)    # (9,)

    return {'mean': mean, 'std': std}

def apply_normalizer(X: List[np.ndarray], stats: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Apply normalization to a dataset using provided mean and std for each digraph matrix in X. 

    Returns a list of new normalized digraph matrices
    """
    mean = stats['mean']
    std = stats['std']

    X_norm = []

    for digraphs in X:
        digraphs_copy = digraphs.copy()

        # Apply ln(1 + x) to time features for normalization 
        for idx in TIME_FEATURES:
            digraphs_copy[:, idx] = np.log1p(np.abs(digraphs_copy[:, idx])) * np.sign(digraphs_copy[:, idx])

        # Normalize using mean and std (ONLY ONCE!)
        digraphs_new = (digraphs_copy - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
        X_norm.append(digraphs_new)
    
    return X_norm