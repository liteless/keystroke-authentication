import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict

# # OPTION 1: coordinates = indexes
# KEYS = [["", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0", "", "", "", "Back"],
#         ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
#         ["", "A", "S", "D", "F", "G", "H", "J", "K", "L"],
#         ["LShiftKey", "Z", "X", "C", "V", "B", "N", "M", "Oemcomma", "OemPeriod", "", "RShiftKey"],
#         ["", "", "", "", "", "Space", "Space", "Space", "Space", "Space"]]

# KEYBOARD_LAYOUT = {}
# for r, row in enumerate(KEYS):
#     for c, key in enumerate(row):
#         KEYBOARD_LAYOUT[key] = (c, r)

#---- STEP 1: DEFINE KEYBOARD LAYOUT ----
# OPTION 2: Manually defined coordinates
KEYBOARD_LAYOUT = {
    #D1 is chosen to be (0, 0) for simplicity
    #Row -1: Escape & Function Keys 
    'Escape': (-1, -1), 'F1': (0, -1), 'F2': (1, -1), 'F3': (2, -1), 'F4': (3, -1),
    'F5': (4, -1), 'F6': (5, -1), 'F7': (6, -1), 'F8': (7, -1), 'F9': (8, -1), 'F10': (9, -1),
    'F11': (10, -1), 'F12': (11, -1),

    #Row 0: Numbers & Backspace
    'Oemtilde': ( -1, 0), 'D1': (0, 0), 'D2': (1, 0), 'D3': (2, 0), 'D4': (3, 0), 'D5': (4, 0),
    'D6': (5, 0), 'D7': (6, 0), 'D8': (7, 0), 'D9': (8, 0), 'D0': (9, 0),
    'OemMinus': (10, 0), 'Oemplus': (11, 0),
    'Back': (12.5, 0), 'Insert': (15, 0), 'Home': (16, 0), 'PageUp': (17, 0),

    #Row 1: QWERTY Row (shifted by 0.5)
    'Tab': (-0.75, 1),
    'Q': (0.5, 1), 'W': (1.5, 1), 'E': (2.5, 1), 'R': (3.5, 1), 'T': (4.5, 1),
    'Y': (5.5, 1), 'U': (6.5, 1), 'I': (7.5, 1), 'O': (8.5, 1), 'P': (9.5, 1),
    'OemOpenBrackets': (10.5, 1), 'OemCloseBrackets': (11.5, 1), 'OemPipe': (13, 1),
    'Delete': (15, 1), 'End': (16, 1), 'PageDown': (17, 1),

    #Row 2: ASDF Row (shifted by 0.75)
    'Capital': (-0.25, 2), 
    'A': (0.75, 2), 'S': (1.75, 2), 'D': (2.75, 2), 'F': (3.75, 2), 'G': (4.75, 2),
    'H': (5.75, 2), 'J': (6.75, 2), 'K': (7.75, 2), 'L': (8.75, 2),
    'OemSemicolon': (9.75, 2), 'OemQuotes': (10.75, 2), 'Return': (12, 2),

    #Row 3: ZXCV row (shifted by 1.25)
    'LShiftKey': (0, 3),
    'Z': (1.25, 3), 'X': (2.25, 3), 'C': (3.25, 3), 'V': (4.25, 3),
    'B': (5.25, 3), 'N': (6.25, 3), 'M': (7.25, 3),
    'Oemcomma': (8.25, 3), 'OemPeriod': (9.25, 3), 'OemQuestion': (10.25, 3),
    'RShiftKey': (10, 3),

    #Row 4: Spacebar (centered) and modifiers
    'LControlKey': (0, 4), 'LMenu': (1, 4), 
    'Space': (5.5, 4),
    'RMenu': (9, 4), 'RControlKey': (11, 4),

    #Numpad coordinates (placed far right)
    'NumPad7': (18, 1), 'NumPad8': (19, 1), 'NumPad9': (20, 1),
    'NumPad4': (18, 2), 'NumPad5': (19, 2), 'NumPad6': (20, 2),
    'NumPad1': (18, 3), 'NumPad2': (19, 3), 'NumPad3': (20, 3),
    'NumPad0': (18.5, 4),

    #Arrow Keys
    'Up': (16, 3), 'Down': (16, 4), 'Left': (15, 4), 'Right': (17, 4)

}

#Some additional keys with multiple names
KEYBOARD_LAYOUT.update({
    "Oem1": KEYBOARD_LAYOUT.get("OemSemicolon"),      # ; :
    "Oem2": KEYBOARD_LAYOUT.get("OemQuestion"),       # / ?
    "Oem3": KEYBOARD_LAYOUT.get("Oemtilde"),          # ` ~
    "Oem4": KEYBOARD_LAYOUT.get("OemOpenBrackets"),   # [ {
    "Oem5": KEYBOARD_LAYOUT.get("OemPipe"),           # \ |
    "Oem6": KEYBOARD_LAYOUT.get("OemCloseBrackets"),  # ] }
    "Oem7": KEYBOARD_LAYOUT.get("OemQuotes"), 
    'Oemminus': KEYBOARD_LAYOUT.get('OemMinus'),
    'OemPlus': KEYBOARD_LAYOUT.get('Oemplus'),
})


#---- STEP 2: CREATE A CLASS TO HOLD KEYSTROKE SESSIONS ---- 
#Helper functions for keystroke analysis 
def get_spatial_distances(key1: str, key2: str):
    """Calculate the signed horizontal and vertical distances from key1 to key2"""
    if key1 not in KEYBOARD_LAYOUT or key2 not in KEYBOARD_LAYOUT:
        print(f"Warning: One of the keys '{key1}' or '{key2}' not found in keyboard layout.")
        return (0.0, 0.0)  # Default to zero distance if key not found
    x1, y1 = KEYBOARD_LAYOUT[key1]
    x2, y2 = KEYBOARD_LAYOUT[key2]

    dist = (x2 - x1, y2 - y1)
    return dist

class KeystrokeSession:
    """Represents a single typing session"""

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.events = []  # List of tuples (timestamp_ms, key_code, event_type)

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
    Extract 9 digraph feature vectors from a keystroke session
        1. Horizontal distance between key2 and key1
        2. Vertical distance between key2 and key1
        3. Euclidean distance between key2 and key1
        4. Key1 hold time (release - press)
        5. Key2 hold time (release - press)
        6. Inner-key time (key2_press - key1_release)
        7. Outer-key time (key2_release - key1_press)
        8. Keydown-to-keydown time (key2_press - key1_press)
        9. Keyup-to-keyup time (key2_release - key1_release)

    Returns:
        numpy array of shape (n_digraphs, 5)
    """
    #Ensure events are sorted by timestamp
    session.sort_events()
    sorted_events = session.events
    
    keypresses = [] #list of keydown order
    active_presses = {}  # dicto of key -> press_time
    
    for event in sorted_events:
        key, ts, etype = event['key'], event['timestamp'], event['event']
        
        #If the event is a key press, store the timestamp
        #If it's a release, find the matching press and record the pair
        #We want to store in order of key presses rather than releases
        if etype == 'press':
            # create record at keydown, append in order
            idx = len(keypresses)
            keypresses.append({
                'key': key,
                'press': ts,
                'release': None,
                'hold_time': None,
            })
            active_presses[key] = idx

        elif etype == 'release' and key in active_presses:
            idx = active_presses.pop(key)
            press_ts = keypresses[idx]['press']
            keypresses[idx]['release'] = ts
            keypresses[idx]['hold_time'] = ts - press_ts
    
    #Keep only the pairs that have a release time recorded
    key_pairs = [kp for kp in keypresses if kp['release'] is not None]

    digraphs = []
    #Iterate through consecutive key pairs to form digraphs
    for i in range(len(key_pairs) - 1):
        k1, k2 = key_pairs[i], key_pairs[i+1]
        if k1['key'] not in KEYBOARD_LAYOUT or k2['key'] not in KEYBOARD_LAYOUT:
            continue  #skip if either key is not in the layout (like LWin or RWin, which are mouse movements)
        distances = get_spatial_distances(k1['key'], k2['key'])
        
        digraphs.append([
            distances[0],
            distances[1],
            np.sqrt(distances[0]**2 + distances[1]**2),
            k1['hold_time'],
            k2['hold_time'],
            k2['press'] - k1['release'],
            k2['release'] - k1['press'],
            k2['press'] - k1['press'],
            k2['release'] - k1['release'],
        ])

    return np.array(digraphs, dtype=np.float32) 


def split_window_digraphs(digraphs: np.ndarray, window_size: int = 50, step: int = 25, min_len: int = 30) -> list:
    """
    Split a digraph sequence into sliding windows.

    Args:
        digraphs: numpy array of shape (L, features)
        window_size: target number of digraphs per window
        step: step size between window starts (controls overlap)
        min_len: minimum acceptable length for a window; windows shorter than
                 this will be discarded.

    Returns:
        List of numpy arrays each with shape (window_len, features)
    """
    windows = []
    if digraphs is None:
        return windows

    L = digraphs.shape[0]
    if L < min_len:
        return windows

    # If entire sequence is shorter than window_size but >= min_len, return single window
    if L <= window_size:
        windows.append(digraphs.copy())
        return windows

    # Sliding windows
    start = 0
    while start < L:
        end = start + window_size
        if end <= L:
            windows.append(digraphs[start:end].copy())
        else:
            # For the final window, include remainder only if it's >= min_len
            if L - start >= min_len:
                windows.append(digraphs[start:L].copy())
            break
        start += step

    return windows

# def normalize_digraphs(digraphs: np.ndarray) -> np.ndarray:
#     """ Normalize digraph features """
#     if len(digraphs) == 0:
#         return digraphs
    
#     normalized = digraphs.copy()

#     # Use log transform for time features to handle wide range of values
#     for i in range(5):
#         normalized[:, i] = np.log1p(normalized[:, i])
        
#     return normalized

#---- STEP 3: PARSE RAW DATA FILES INTO KEYSTROKE SESSIONS ----
def parse_event_type(raw: str) -> str | None: 
    """Convert raw event type to 'press' or 'release'"""
    if raw == 'KeyDown':
        return 'press'
    elif raw == 'KeyUp':
        return 'release'
    else:
        return None

def load_all_sessions(root_dir: str) -> List[KeystrokeSession]:
    """Go through s0, s1, and s2 under the root directory, parsing all task=0 files into KeystrokeSession objects.
    Returns: flat list of KeystrokeSession objects
    """
    root = Path(root_dir)
    sessions: List[KeystrokeSession] = []

    for split_name in ['s0', 's1', 's2']:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue

        for path in split_dir.rglob("*.txt"):
            #Expects a step of the form WWWXYZ, where WWW is the user ID, X is the session ID, Y is the keyboard type, and Z=0 is the task ID
            stem = path.stem

            #Skip non-task 0 files or incorrectly named files
            if len(path.stem) != 6 or path.stem[-1] != '0':
                continue
            
            user_id = int(stem[:3])
            session_id = int(stem[3])
            task_id = int(stem[5])
            
            #Extra check to ensure we only have task 0 files
            if task_id != 0:
                continue

            session = KeystrokeSession(user_id=user_id, session_id=session_id)

            #Extract the actual keystroke events from each session file
            with path.open('r', encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) != 3: #Check for correct formatting
                        continue
                    
                    key_name, event, timestamp = parts[0], parts[1], parts[2]
                    event = parse_event_type(event)

                    #Extra check to ensure it is a valid event (KeyDown or KeyUp)
                    if event is None:
                        continue 
                    
                    #Try to parse timestamp as an integer
                    try:
                        timestamp = int(timestamp)
                    except ValueError:
                        continue
                    
                    session.add_event(
                        timestamp=timestamp,
                        key_code=key_name,
                        event_type=event
                    )

            session.sort_events()
            if session.events: 
                sessions.append(session)
    
    return sessions

#---- STEP 4: CREATE TRAIN AND TEST DATA BY LOOPING OVER PARSED DATASET ----
def build_datasets(root_dir: str, window_size: int = 50, step: int = 25, min_len: int = 30) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Load all sessions from the root directory, extract the digraphs, and split into
    training (users 001-100) and testing (users 101-148)
    
    L_i = number of digraphs for user i 
    Returns: 
        X_train: list of (L_i, 9) digraph arrays  
        y_train: list of user IDs corresponding to each array in X_train
        X_test: list of (L_j, 9) digraph arrays
        y_test: list of user IDs corresponding to each array in X_test
        user_sessions_train: dict mapping user ID to their correponding session indices in X_train
        user_sessions_test: dict mapping user ID to their corresponding session indices in X_test
        """
    sessions = load_all_sessions(root_dir)

    X_train: List[np.ndarray] = []
    y_train: List[int] = []
    X_test: List[np.ndarray] = []
    y_test: List[int] = []

    user_sessions_train: Dict[int, List[int]] = defaultdict(list)
    user_sessions_test: Dict[int, List[int]] = defaultdict(list)

    for _, session in enumerate(sessions): 
        #extract digraph for this session
        digraphs = extract_digraphs(session) #(L_i, 9)

        #Ensure there are no empty sessions and only valid digraphs
        if digraphs is None or digraphs.shape[0] == 0:
            continue

        # Split session digraphs into smaller windows
        windows = split_window_digraphs(digraphs, window_size=window_size, step=step, min_len=min_len)

        if not windows:
            continue

        user_id = session.user_id

        # Add each window as its own sample; update user_sessions mapping accordingly
        for win in windows:
            if 1 <= user_id <= 100:
                session_idx = len(X_train)
                X_train.append(win)
                y_train.append(user_id)
                user_sessions_train[user_id].append(session_idx)
            else:
                session_idx = len(X_test)
                X_test.append(win)
                y_test.append(user_id)
                user_sessions_test[user_id].append(session_idx)
    
    #Convert y_train and y_test to numpy arrays
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test, user_sessions_train, user_sessions_test


    



