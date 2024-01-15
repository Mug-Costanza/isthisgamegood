import steamreviews
import re
import requests
from bs4 import BeautifulSoup
import json
import csv
import random

def search_steam_game(game_title):
    search_url = f"https://store.steampowered.com/search/?term={game_title.replace(' ', '+')}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        search_results = soup.find_all('a', class_='search_result_row')
        
        if search_results:
            # Get the URL of the first search result (assuming it's the game page)
            game_url = search_results[0]['href']
            app_id_match = re.search(r'/app/(\d+)/', game_url)
            if app_id_match:
                return app_id_match.group(1)
            else:
                return None
        else:
            return None
    else:
        return None
    
def fetch_reviews(app_id, num_reviews):
    reviews = []
    cursor = "*"
    base_url = f"https://store.steampowered.com/appreviews/{app_id}"

    while len(reviews) < num_reviews:
        params = {
            "json": "1",
            "language": "english",
            "filter": "all",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": min(num_reviews - len(reviews), 100),
            "cursor": cursor
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            reviews.extend(data["reviews"])
            cursor = data.get("cursor", "*")  # Update the cursor for the next request
        else:
            print("Failed to fetch reviews.")
            break

    return reviews[:num_reviews]  # Return only the specified number of reviews

def save_reviews(app_id, reviews):
    filename = f"reviews_{app_id}.json"
    with open(filename, "w") as file:
        json.dump(reviews, file, indent=4)
    print(f"Downloaded {len(reviews)} reviews. Saved to '{filename}'.")
    
def save_reviews_as_csv(app_id, reviews):
    filename = f"reviews_{app_id}.csv"
    headers = reviews[0].keys() if reviews else []
    with open(filename, "w", newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"Downloaded {len(reviews)} reviews. Saved to '{filename}'.")

def print_random_review(reviews):
    if reviews:
        random_review = random.choice(reviews)
        print(f"\nRandom Review:")
        print(f"Author: {random_review['author']}")
        print(f"Review: {random_review['review']}")
    else:
        print("No reviews available.")

def main():
    app_id_input = input("Enter the Game Title: ")
    app_id = search_steam_game(app_id_input)
    num_reviews = int(input("Enter the number of reviews to download: "))

    fetched_reviews = fetch_reviews(app_id, num_reviews)

    if fetched_reviews:
        save_reviews_as_csv(app_id, fetched_reviews)
        print_random_review(fetched_reviews)
    else:
        print("No reviews fetched.")

if __name__ == "__main__":
    main()
