from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lib.hard_coded_constants import NEWS_CATEGORIES

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


# writing a more general function for classifying using models for my use case to try and avoid code redundancy
def general_headline_classifier(model_name, category_name, headlines, indexes, classification_categories=NEWS_CATEGORIES, batch_size=32, is_zeroshot=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_model.to(device)
    print(f"running on device: {device}")
    classifier_model.eval()

    num_batches = (len(headlines) + batch_size - 1) // batch_size
    print(f"processing {len(headlines)} headlines in {num_batches} batches")

    classified_dfs = []

    for i in range(num_batches):
        print(f"starting batch {i + 1}/{num_batches}")

        start = i * batch_size
        end = min(start + batch_size, len(headlines))
        batch_headlines = headlines[start:end]
        batch_indexes = indexes[start:end]

        # Repeat each headline for all categories if zeroshot classification
        if is_zeroshot:
            repeated_headlines = np.repeat(batch_headlines, len(classification_categories)).tolist()
            repeated_categories = classification_categories * len(batch_headlines)

            inputs = tokenizer(
                repeated_headlines,
                repeated_categories,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            #extract probabilities for each category
            with torch.no_grad():
                logits = classifier_model(**inputs).logits

            entailment_scores = logits[:, 2]
            entailment_scores = entailment_scores.view(len(batch_headlines), len(classification_categories))
            probs = F.softmax(entailment_scores, dim=1)

            # Convert the tensor to a numpy array
            probs_np = probs.cpu().numpy()
        else:
            inputs = tokenizer(
                batch_headlines,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            #extract probabilities of each category
            with torch.no_grad():
                logits = classifier_model(**inputs).logits

            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()
            classification_categories = [classifier_model.config.id2label[i] for i in range(classifier_model.config.num_labels)]

        # Create dataframe
        df = pd.DataFrame(probs_np, columns=classification_categories)
        df[category_name] = df[classification_categories].idxmax(axis=1)
        # Add the headlines as a column
        df["headline"] = batch_headlines
        df["index"] = batch_indexes

        classified_dfs.append(df)

    # Concatenate all dfs
    combined_classified_df = pd.concat(classified_dfs)

    return combined_classified_df


# handy function that helps join subcolumns to the big data csv
# process is: find sentiment scores/emotion scores/classification of headlines on subset of headlines
# these headlines already exist in the big data csv, how should I add in this new subset with additional info?
# either delete existing rows then add new rows with the additional info
# or this approach, map the additional info from the subset of headlines into the big data csv dataframe
def map_values_into_data_df(data_df, classified_df, col_to_map):
    # Create a Series from the mapping: index -> value
    mapped_series = pd.Series(classified_df[col_to_map].values, index=classified_df["index"])
    # create col if it doesn't exist (like if we are just starting out and creating all the data)
    if col_to_map not in data_df.columns:
        data_df[col_to_map] = None
    # keep existing values in the relevant column, map in new additional info
    data_df[col_to_map] = mapped_series.reindex(data_df.index).combine_first(data_df[col_to_map])

    return data_df