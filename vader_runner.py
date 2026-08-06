"""Run VADER on reviews_clean.csv and save."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nlp_analysis import apply_vader
from src.utils import PROCESSED_DIR
import pandas as pd

start = time.time()
df = pd.read_csv(f'{PROCESSED_DIR}/reviews_clean.csv')
print(f'Loaded {len(df)} reviews', flush=True)

df = apply_vader(df)
df.to_csv(f'{PROCESSED_DIR}/reviews_clean.csv', index=False)

elapsed = time.time() - start
print(f'VADER saved in {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)
