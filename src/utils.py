import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SAMPLE_DIR = os.path.join(DATA_DIR, 'sample')

GAMES = {
    1245620: 'Elden Ring',
    1086940: "Baldur's Gate 3",
    730: 'Counter-Strike 2',
    2357570: 'Overwatch 2',
    2767030: 'Marvel Rivals',
    1091500: 'Cyberpunk 2077',
    275850: 'No Man\'s Sky',
     578080: 'PUBG: BATTLEGROUNDS',
    1938090: 'Call of Duty HQ',
    553850: 'Helldivers 2',
    292030: 'The Witcher 3',
    489830: 'Skyrim SE',
    2054970: "Dragon's Dogma 2",
    377160: 'Fallout 4',
    1172470: 'Apex Legends',
    1085660: 'Destiny 2',
    359550: 'Rainbow Six Siege',
    1517290: 'Battlefield 2042',
    440: 'Team Fortress 2',
    444090: 'Paladins',
    1097150: 'Fall Guys',
    2215430: 'Ghost of Tsushima',
    1593500: 'God of War',
    1174180: 'Red Dead Redemption 2',
     601150: 'Devil May Cry 5',
    281990: 'Stellaris',
    289070: 'Sid Meier\'s Civilization VI',
    1142710: 'Total War: WARHAMMER III',
    1466860: 'Age of Empires IV',
    252490: 'Rust',
    346110: 'ARK: Survival Evolved',
    892970: 'Valheim',
    242760: 'The Forest',
}

GAME_IDS = list(GAMES.keys())

GENRES = {
    'RPG': [1086940, 1091500, 1245620, 292030, 489830, 2054970, 377160, 892970],
    'Shooter': [730, 578080, 553850, 1938090, 2357570, 1172470, 1085660, 359550, 1517290, 440, 444090],
    'Hero_Shooter': [2357570, 2767030, 440, 444090],
    'Battle_Royale': [578080, 1938090, 1172470, 1097150],
    'Action': [1245620, 1091500, 275850, 553850, 2215430, 1593500, 1174180, 601150, 292030, 489830, 2054970, 377160, 440, 252490, 346110, 892970, 242760],
    'Strategy': [1086940, 281990, 289070, 1142710, 1466860, 359550],
    'Adventure': [1086940, 275850, 292030, 489830, 377160, 2215430, 1593500, 1174180, 346110, 892970, 242760],
    'Survival': [275850, 252490, 346110, 892970, 242760],
    'Free_to_Play': [730, 2357570, 2767030, 578080, 1172470, 440, 444090, 1097150],
}

GENRE_IDS = list(GENRES.keys())

GAME_GENRES = {}
for genre, ids in GENRES.items():
    for app_id in ids:
        GAME_GENRES.setdefault(app_id, []).append(genre)

def games_by_genre(genre):
    return GENRES.get(genre, [])

def games_in_genres(genre_list):
    ids = set()
    for g in genre_list:
        ids.update(GENRES.get(g, []))
    return sorted(ids)

def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR]:
        os.makedirs(d, exist_ok=True)

def safe_request(url, params=None, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            time.sleep(delay * (attempt + 1))
        except requests.RequestException:
            time.sleep(delay * (attempt + 1))
    return None

def timestamp_to_date(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


def load_raw(filename):
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
