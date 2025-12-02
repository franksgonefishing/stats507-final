from lib.hard_coded_constants import NEWS_CATEGORIES, DATA_FILE_NAME, NEWS_CATEGORY_CLASSIFICATION_MODEL
from lib.eval_classifier_model import general_headline_classifier, map_values_into_data_df
from lib.sentiment_analysis import find_filtered_headlines_and_indices

import pandas as pd


# to be run after headlines are scraped to categorize the specific news category
# analysis is primarily done on headlines classified as "politics"
def classify_headlines(news_site, year, month, day=None, batch_size=32):
    print(f"classifying {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")

    complete_df = pd.read_csv(DATA_FILE_NAME, low_memory=False)
    
    headlines, selected_indices = find_filtered_headlines_and_indices(complete_df, news_site, year, month, day)
    
    if len(headlines) == 0:
        print(f"found no headlines for {news_site} on {year}-{month}{'-' + str(day) if day else ''}")
        return

    combined_classified_df = general_headline_classifier(
        NEWS_CATEGORY_CLASSIFICATION_MODEL, 
        "news_category", 
        headlines, 
        selected_indices,
        batch_size=batch_size
    )

    # For each news category, align by index and fill missing values with original column values
    for cat in NEWS_CATEGORIES:
        complete_df = map_values_into_data_df(complete_df, combined_classified_df, cat)
    # one more time for overall category based on highest probability
    complete_df = map_values_into_data_df(complete_df, combined_classified_df, "news_category")

    print(f"Finished processing {len(headlines)} headlines")
    
    complete_df.to_csv(DATA_FILE_NAME, index=False)