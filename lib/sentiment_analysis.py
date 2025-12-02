from nltk.sentiment import SentimentIntensityAnalyzer
from lib.hard_coded_constants import DATA_FILE_NAME, EMOTION_ANALYSIS_CLASSIFICATION_MODEL, EMOTION_CATEGORIES, GENERATED_HEADLINES_FILE_NAME
from lib.eval_classifier_model import map_values_into_data_df, general_headline_classifier

import torch.nn.functional as F
import pandas as pd


def find_pos_neg_neu_sentiment(news_site, year, month, day=None, use_generated_headlines=False):
    if use_generated_headlines:
        data_set = GENERATED_HEADLINES_FILE_NAME
        year = 2025
        month = 10
        day = 10
    else:
        data_set = DATA_FILE_NAME

    print(f"finding pos_neg_neu sentiment for {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")
    # relative path that works when run by the file in final_project
    complete_df = pd.read_csv(data_set, low_memory=False)
    # Base conditions
    mask = (complete_df["network"] == news_site) & (complete_df["year"] == year) & (complete_df["month"] == month)
    # Add the optional condition only if day exists
    if day:
        mask &= (complete_df["day"] == day)

    selected_indices = complete_df[mask].index.to_list()
    headlines = complete_df[mask]["headline"].to_list()

    if len(headlines) == 0:
        print(f"found no headlines for {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")
        return
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    df_dict = {
        "headline": [],
        "index": [],
        "vader_compound_score": []
    }

    for headline, index in zip(headlines, selected_indices):
        sentiment_score = sid.polarity_scores(headline)

        df_dict["headline"].append(headline)
        df_dict["index"].append(index)
        df_dict["vader_compound_score"].append(sentiment_score["compound"])
    
    sentiment_df = pd.DataFrame(df_dict)

    complete_df = map_values_into_data_df(complete_df, sentiment_df, "vader_compound_score")

    complete_df.to_csv(data_set, index=False)

    print(f"added {len(headlines)} pos_neg_neu sentiments for {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")


def find_emotion_sentiment(news_site, year, month, day=None, use_generated_headlines=False):
    if use_generated_headlines:
        data_set = GENERATED_HEADLINES_FILE_NAME
    else:
        data_set = DATA_FILE_NAME

    print(f"finding emotion sentiment for {news_site} headlines on {year}-{month}{'-' + str(day) if day else ''}")
    # relative path that works when run by the file in final_project
    complete_df = pd.read_csv(data_set, low_memory=False)
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
        EMOTION_ANALYSIS_CLASSIFICATION_MODEL, 
        "primary_emotion", 
        headlines, 
        selected_indices,
        is_zeroshot=False
    )
    #also want to find secondary emotion
    combined_classified_df["secondary_emotion"] = (
        combined_classified_df[EMOTION_CATEGORIES]
        .apply(lambda row: row.nlargest(2).index[1], axis=1)
    )
    # For each category, align by index and fill missing values with original column values
    for cat in EMOTION_CATEGORIES:
        complete_df = map_values_into_data_df(complete_df, combined_classified_df, cat)
    # one more time for overall category based on highest probability
    complete_df = map_values_into_data_df(complete_df, combined_classified_df, "primary_emotion")
    complete_df = map_values_into_data_df(complete_df, combined_classified_df, "secondary_emotion")

    print(f"Finished processing {len(headlines)} headlines")
    
    complete_df.to_csv(data_set, index=False)