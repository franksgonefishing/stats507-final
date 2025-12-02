from lib.hard_coded_constants import NEWS_CATEGORIES, DATA_FILE_NAME, NEWS_CATEGORY_CLASSIFICATION_MODEL
from lib.eval_classifier_model import general_headline_classifier, map_values_into_data_df

import pandas as pd

def classify_headlines(news_site, year, month, day=None, batch_size=32):
    print(f"classifying {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")
    # relative path that works when run by the file in final_project
    complete_df = pd.read_csv(DATA_FILE_NAME, low_memory=False)
    # Base conditions
    mask = (complete_df["network"] == news_site) & (complete_df["year"] == year) & (complete_df["month"] == month)
    # Add the optional condition only if day exists
    if day:
        mask &= (complete_df["day"] == day)

    selected_indices = complete_df[mask].index.to_list()
    headlines = complete_df[mask]["headline"].to_list()
    
    if len(headlines) == 0:
        print(f"found no headlines for {news_site} on {year}-{month}{'-' + str(day) if day else ''}")
        return

    # Concatenate all classified DataFrames
    combined_classified_df = general_headline_classifier(
        NEWS_CATEGORY_CLASSIFICATION_MODEL, 
        "news_category", 
        headlines, 
        selected_indices,
    )

    # For each category, align by index and fill missing values with original column values
    for cat in NEWS_CATEGORIES:
        complete_df = map_values_into_data_df(complete_df, combined_classified_df, cat)
    # one more time for overall category based on highest probability
    complete_df = map_values_into_data_df(complete_df, combined_classified_df, "news_category")

    print(f"Finished processing {len(headlines)} headlines")
    
    complete_df.to_csv(DATA_FILE_NAME, index=False)