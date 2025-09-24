from google_play_scraper import reviews_all, Sort
import pandas as pd
import csv

class AppReviewScraper:
    def __init__(self, app_id, lang='id', country='id', sort=Sort.MOST_RELEVANT, count=15000):
        self.app_id = app_id
        self.lang = lang
        self.country = country
        self.sort = sort
        self.count = count
        self.reviews = []

    def fetch_reviews(self):
        self.reviews = reviews_all(
            self.app_id,
            lang=self.lang,
            country=self.country,
            sort=self.sort,
            count=self.count
        )

    def save_to_csv(self, filename):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['review'])
            for review in self.reviews:
                writer.writerow([review['content']])

    def load_to_dataframe(self):
        return pd.DataFrame(self.reviews)

if __name__ == "__main__":
    scraper = AppReviewScraper('com.bareksa.app')
    scraper.fetch_reviews()
    scraper.save_to_csv('ulasan_aplikasi.csv')
    app_reviews_df = scraper.load_to_dataframe()
    print(app_reviews_df.shape)
    print(app_reviews_df.head())
    app_reviews_df.to_csv('ulasan_aplikasi.csv', index=False)