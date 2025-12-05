import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# OPTION 1: coordinates = indexes
KEYS = [["", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0", "", "", "", "Back"],
        ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["", "A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["LShiftKey", "Z", "X", "C", "V", "B", "N", "M",
            "Oemcomma", "OemPeriod", "", "RShiftKey"],
        ["", "", "", "", "", "Space", "Space", "Space", "Space", "Space"]]

KEYBOARD_LAYOUT = {}
for r, row in enumerate(KEYS):
    for c, key in enumerate(row):
        KEYBOARD_LAYOUT[key] = (c, r)

# OPTION 2: Manually defined coordinates
KEYBOARD_LAYOUT = {
    # --- Row 0: Number row ---
    'D1': (0, 0), 'D2': (1, 0), 'D3': (2, 0), 'D4': (3, 0), 'D5': (4, 0),
    'D6': (5, 0), 'D7': (6, 0), 'D8': (7, 0), 'D9': (8, 0), 'D0': (9, 0),
    'Back': (10.5, 0),

    # --- Row 1: QWERTY row ---
    'Q': (0, 1), 'W': (1, 1), 'E': (2, 1), 'R': (3, 1), 'T': (4, 1),
    'Y': (5, 1), 'U': (6, 1), 'I': (7, 1), 'O': (8, 1), 'P': (9, 1),

    # --- Row 2: ASDF row (shifted) ---
    'A': (0.5, 2), 'S': (1.5, 2), 'D': (2.5, 2), 'F': (3.5, 2), 'G': (4.5, 2),
    'H': (5.5, 2), 'J': (6.5, 2), 'K': (7.5, 2), 'L': (8.5, 2),

    # --- Row 3: ZXCV row (shifted more) ---
    'LShiftKey': (0, 3),
    'Z': (1, 3), 'X': (2, 3), 'C': (3, 3), 'V': (4, 3),
    'B': (5, 3), 'N': (6, 3), 'M': (7, 3),
    'Oemcomma': (8, 3), 'OemPeriod': (9, 3),
    'RShiftKey': (10, 3),

    # --- Row 4: Spacebar row ---
    'Space': (4.5, 4),

    # --- Numpad coordinates ---
    'NumPad7': (14, 1), 'NumPad8': (15, 1), 'NumPad9': (16, 1),
    'NumPad4': (14, 2), 'NumPad5': (15, 2), 'NumPad6': (16, 2),
    'NumPad1': (14, 3), 'NumPad2': (15, 3), 'NumPad3': (16, 3),
    'NumPad0': (15, 4),
}


def get_spatial_distances(key1: str, key2: str):
    """Calculate the signed horizontal and vertical distances from key1 to key2"""
    if key1 not in KEYBOARD_LAYOUT or key2 not in KEYBOARD_LAYOUT:
        return 0.0
    x1, y1 = KEYBOARD_LAYOUT[key1]
    x2, y2 = KEYBOARD_LAYOUT[key2]

    dist = (x2 - x1, y2 - y1)
    return dist


class KeystrokeSession:
    """Represents a single typing session"""

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.events = []  # List of (timestamp_ms, key_code, event_type)

    # NOTE: try to parallelize this part later
    def add_event(self, timestamp: float, key_code: str, event_type: str):
        """Add a keystroke event ('press' or 'release')"""
        self.events.append({
            'timestamp': timestamp,
            'key': key_code,
            'event': event_type
        })

    def sort_events(self):
        """Sort events by timestamp"""
        self.events.sort(key=lambda x: x['timestamp'])


def extract_digraphs(session: KeystrokeSession) -> np.ndarray:
    """
    Extract 8 digraph feature vectors from a keystroke session
        1. Horizontal distance between key2 and key1
        2. Vertical distance between key2 and key1
        3. Euclidean distance between key2 and key1
        4. Key1 hold time (release - press)
        5. Key2 hold time (release - press)
        6. Outer-key time (key2_press - key1_release)
        7. Inner-key time (key2_release - key1_press)
        8. Keydown-to-keydown time (key2_press - key1_press)
        9. Keyup-to-keyup time (key2_release - key1_release)

    Returns:
        numpy array of shape (n_digraphs, 5)
    """
    sorted_events = session.sort_events().events

    key_pairs = []
    active_presses = {}  # key -> press_time

    for event in sorted_events:
        key, ts, etype = event['key'], event['timestamp'], event['event']

        if etype == 'press':
            active_presses[key] = ts
        elif etype == 'release' and key in active_presses:
            press_ts = active_presses.pop(key)
            key_pairs.append({
                'key': key,
                'press': press_ts,
                'release': ts,
                'hold_time': ts - press_ts
            })

    digraphs = []
    for i in range(len(key_pairs) - 1):
        k1, k2 = key_pairs[i], key_pairs[i+1]
        distances = get_spatial_distances(k1['key'], k2['key'])

        digraphs.append({
            'spatial_x': distances[0],
            'spatial_y': distances[1],
            'spatial_dist': np.sqrt(distances[0]**2 + distances[1]**2),

            'hold1': k1['hold_time'],
            'hold2': k2['hold_time'],

            'outer-time': k2['press'] - k1['release'],
            'inner-time': k2['release'] - k1['press'],
            'down_to_down': k2['press'] - k1['press'],
            'up_to_up': k2['release'] - k1['release'],
        })

    return np.array(digraphs)


def normalize_digraphs(digraphs: np.ndarray) -> np.ndarray:
    """ Normalize digraph features """
    if len(digraphs) == 0:
        return digraphs

    normalized = digraphs.copy()

    # Use log transform for time features to handle large values
    for i in range(3, 5):
        normalized[:, i] = np.log1p(normalized[:, i])

    # Standardize all feature
    for i in range(5):
        feature_values = normalized[:, i]
        if feature_values.std() > 0:
            normalized[:, i] = (
                feature_values - feature_values.mean()) / feature_values.std()

    return normalized
