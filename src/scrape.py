import time
import pandas as pd
from src.utils import (
    safe_request, GAMES, GAME_IDS, RAW_DIR, ensure_dirs
)

STEAM_APP_DETAILS_URL = 'https://store.steampowered.com/api/appdetails'
STEAM_REVIEWS_URL = 'https://store.steampowered.com/appreviews'
STEAM_PLAYER_COUNT_URL = 'https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/'


def fetch_app_details(app_id):
    data = safe_request(STEAM_APP_DETAILS_URL, {'appids': app_id})
    if data and str(app_id) in data:
        d = data[str(app_id)].get('data', {})
        return {
            'app_id': app_id,
            'name': d.get('name', ''),
            'release_date': d.get('release_date', {}).get('date', '') if d.get('release_date') else '',
            'developers': ', '.join(d.get('developers', [])),
            'publishers': ', '.join(d.get('publishers', [])),
            'genres': ', '.join(g['description'] for g in d.get('genres', [])),
            'categories': ', '.join(c['description'] for c in d.get('categories', [])),
            'price': d.get('price_overview', {}).get('final', 0) if d.get('price_overview') else 0,
            'metacritic_score': d.get('metacritic', {}).get('score', None) if d.get('metacritic') else None,
            'recommendations': d.get('recommendations', {}).get('total', 0) if d.get('recommendations') else 0,
        }
    return None


def scrape_reviews(app_id, max_pages=50, reviews_per_page=100):
    all_reviews = []
    cursor = '*'
    seen_ids = set()
    empty_page_count = 0

    for page in range(max_pages):
        params = {
            'json': 1,
            'language': 'english',
            'num_per_page': reviews_per_page,
            'purchase_type': 'all',
            'day_range': 9999,
            'cursor': cursor,
        }
        data = safe_request(f'{STEAM_REVIEWS_URL}/{app_id}', params)
        if not data or not data.get('success'):
            break

        reviews = data.get('reviews', [])
        if not reviews:
            break

        new_count = 0
        for r in reviews:
            rid = r.get('recommendationid', '')
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            new_count += 1
            all_reviews.append({
                'app_id': app_id,
                'game_name': GAMES.get(app_id, ''),
                'review_id': rid,
                'author_id': r.get('author', {}).get('steamid', ''),
                'num_games_owned': r.get('author', {}).get('num_games_owned', 0),
                'num_reviews': r.get('author', {}).get('num_reviews', 0),
                'playtime_forever': r.get('author', {}).get('playtime_forever', 0),
                'playtime_at_review': r.get('author', {}).get('playtime_at_review', 0),
                'language': r.get('language', ''),
                'review_text': r.get('review', ''),
                'timestamp_created': r.get('timestamp_created', 0),
                'timestamp_updated': r.get('timestamp_updated', 0),
                'voted_up': r.get('voted_up', False),
                'votes_up': r.get('votes_up', 0),
                'votes_funny': r.get('votes_funny', 0),
                'weighted_vote_score': r.get('weighted_vote_score', ''),
                'steam_purchase': r.get('steam_purchase', False),
                'received_for_free': r.get('received_for_free', False),
                'written_during_early_access': r.get('written_during_early_access', False),
            })

        if new_count == 0:
            empty_page_count += 1
            if empty_page_count >= 3:
                break
        else:
            empty_page_count = 0

        cursor = data.get('cursor', '*')
        time.sleep(0.3)

    return all_reviews


def fetch_player_count(app_id):
    data = safe_request(STEAM_PLAYER_COUNT_URL, {'appid': app_id, 'format': 'json'})
    if data and 'response' in data:
        return {
            'app_id': app_id,
            'game_name': GAMES.get(app_id, ''),
            'player_count': data['response'].get('player_count', 0),
            'game_id': data['response'].get('game_id', app_id),
        }
    return None


def scrape_all(limit_reviews_per_game=None):
    ensure_dirs()

    app_details = []
    for app_id in GAME_IDS:
        details = fetch_app_details(app_id)
        if details:
            app_details.append(details)
            print(f"  Details: {details['name']}")
        time.sleep(0.3)

    df_details = pd.DataFrame(app_details)
    df_details.to_csv(f'{RAW_DIR}/app_details.csv', index=False)
    print(f"\nSaved app details for {len(app_details)} games")

    all_reviews = []
    for app_id in GAME_IDS:
        name = GAMES[app_id]
        print(f"\nScraping reviews for {name} (app_id={app_id})...")
        game_reviews = scrape_reviews(app_id)
        if limit_reviews_per_game and len(game_reviews) > limit_reviews_per_game:
            game_reviews = game_reviews[:limit_reviews_per_game]
        all_reviews.extend(game_reviews)
        print(f"  Got {len(game_reviews)} reviews")

    df_reviews = pd.DataFrame(all_reviews)
    df_reviews.to_csv(f'{RAW_DIR}/reviews.csv', index=False)
    print(f"\nSaved {len(df_reviews)} total reviews")

    player_snapshots = []
    for app_id in GAME_IDS:
        pc = fetch_player_count(app_id)
        if pc:
            player_snapshots.append(pc)
            print(f"  {GAMES[app_id]}: {pc['player_count']:,} players")
        time.sleep(0.3)

    df_players = pd.DataFrame(player_snapshots)
    df_players.to_csv(f'{RAW_DIR}/player_counts.csv', index=False)
    print(f"\nSaved player counts for {len(player_snapshots)} games")

    return df_details, df_reviews, df_players


if __name__ == '__main__':
    scrape_all()
