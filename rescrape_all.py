import sys
import time
import pandas as pd
from src.utils import GAMES, GAME_IDS, RAW_DIR, PROCESSED_DIR, ensure_dirs
from src.scrape import scrape_reviews
from src.clean import process_reviews
from src.nlp_analysis import apply_vader

OUTPUT = f'{RAW_DIR}/reviews.csv'


def rescrape_all(max_pages=50):
    ensure_dirs()
    total = 0
    first = True

    for i, app_id in enumerate(GAME_IDS, 1):
        name = GAMES[app_id]
        print(f'\n[{i}/{len(GAME_IDS)}] {name} (app_id={app_id})...', flush=True)

        reviews = scrape_reviews(app_id, max_pages=max_pages)
        if not reviews:
            print(f'  No reviews found, skipping')
            continue

        df = pd.DataFrame(reviews)
        mode = 'w' if first else 'a'
        header = first
        df.to_csv(OUTPUT, mode=mode, header=header, index=False)
        first = False

        total += len(df)
        print(f'  Saved {len(df)} reviews (running total: {total})')
        time.sleep(0.5)

    print(f'\nDone! Total reviews saved: {total}')
    return total


def post_process():
    print('\nRunning process_reviews()...')
    df, stats = process_reviews()
    print(f'Clean reviews: {len(df)}')

    print('\nRunning VADER...')
    df = apply_vader(df)
    out_path = f'{PROCESSED_DIR}/reviews_clean.csv'
    df.to_csv(out_path, index=False)
    print(f'VADER saved to {out_path}')

    return df


if __name__ == '__main__':
    total = rescrape_all(max_pages=50)
    if total > 0:
        post_process()
