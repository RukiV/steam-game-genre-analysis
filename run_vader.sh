#!/bin/bash
cd /mnt/c/Users/V/Desktop/Data\ Analise\ Project
source venv/bin/activate
python3 -u -c "
import sys; sys.path.insert(0,'/mnt/c/Users/V/Desktop/Data Analise Project')
import time; start = time.time()
from src.nlp_analysis import apply_vader
from src.utils import PROCESSED_DIR
import pandas as pd
df = pd.read_csv(f'{PROCESSED_DIR}/reviews_clean.csv')
print(f'Loaded {len(df)} reviews', flush=True)
df = apply_vader(df)
df.to_csv(f'{PROCESSED_DIR}/reviews_clean.csv', index=False)
elapsed = time.time() - start
print(f'VADER saved in {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)
" > /tmp/vader_run.log 2>&1
