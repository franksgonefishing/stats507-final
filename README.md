# STATS 507: Final Project - huafrank
The final project for my class STATS 507: Data Science and Analytics using Python (DATASCI 507 from Winter 2026 onwards). This project gathers headline data from many different news networks, analyzes this data in three separate ways, and fine-tunes a causal language model for each news network to generate new headlines in the writing style of each of the news networks.

## Required Packages
Please ensure that the following packages are installed for everything to run properly:
* bs4
* dash
* datasets
* huggingface_hub
* nltk
* numpy
* pandas
* playwright
* plotly
* prince
* scipy
* sklearn
* torch
* transformers

## Running the Tools
Please see below for instructions for each tool. Please run everything from the stats507-final directory.

### Scraper
Runs the scraper that adds headlines to data/headlines_data.csv. <br> <br>
python run_scraper.py <br> <br>
args:
* --sites (required) Enter ALL to run for all sites, otherwise enter sites found in lib.hard_coded_constants.NEWS_SITES_BASE_URL.keys() separated by a space
* --year (required) Enter an int between 2001 and 2025
* --month (required) Enter an int between 1 and 12
* --day [OPTIONAL] Enter an int to scrape a specific day, otherwise all days in the month will be scraped
* --ignore_sites [OPTIONAL] Enter sites to ignore scraping, often used in conjunction will --sites ALL
* --delete_headlines [OPTIONAL] Enter True to delete rows for the specified inputs, avoids having to do this in the CSV


example:
* python run_scraper.py --sites ALL --year 2025 --month 11
* python run_scraper.py --sites Fox NYT --year 2025 --month 6 --day 29

### Classifier
Runs the classifier tool to classify scraped headlines into ["politics", "sports", "business", "entertainment", "technology", "health"]. <br> <br>
python run_classifier.py <br> <br>
args:
* --sites (required) Enter ALL to run for all sites, otherwise enter sites found in lib.hard_coded_constants.NEWS_SITES_BASE_URL.keys() separated by a space
* --year (required) Enter an int between 2001 and 2025
* --month (required) Enter an int between 1 and 12
* --day [OPTIONAL] Enter an int to run for a specific day, otherwise all days in the month will be run
* --ignore_sites [OPTIONAL] Enter sites to ignore during the run, often used in conjunction will --sites ALL


example:
* python run_classifier.py --sites ALL --year 2025 --month 11
* python run_classifier.py --sites ALL --year 2025 --month 10 --day 18 --ignore-sites Fox MSNBC

### Sentiment
Runs the sentiment tool that provides the VADER sentiment score, emotion sentiment classification, or both for headlines or generated headlines. <br> <br>
python run_sentiment.py <br> <br>
args:
* --analysis_type (required) Enter ALL to run both VADER sentiment scoring and emotion sentiment classifier, otherwise enter pos_neg_neu or emotion
* --sites (required) Enter ALL to run for all sites, otherwise enter sites found in lib.hard_coded_constants.NEWS_SITES_BASE_URL.keys() separated by a space
* --year (required) Enter an int between 2001 and 2025
* --month (required) Enter an int between 1 and 12
* --day [OPTIONAL] Enter an int to run for a specific day, otherwise all days in the month will be run
* --ignore_sites [OPTIONAL] Enter sites to ignore during the run, often used in conjunction will --sites ALL
* --use_generated_headlines [OPTIONAL] Enter True to run this on headlines generated from the language models


example:
* python run_sentiment.py --analysis_type ALL --sites ALL --year 2025 --month 11
* python run_sentiment.py --analysis_type emotion --sites NYT --year 2025 --month 10 --use_generated_headlines True

### Dashboard
Kicks off the dashboard for viewing the data and analyses. Please make sure all the packages above are installed. <br> <br>
python run_dashboard.py <br> <br>
args:
* None


example:
* python run_dashboard.py
* Look for a message similar to: 'Dash is running on http://127.0.0.1:8050/'

### fine_tune_language_model.ipynb
Notebook that fine-tunes language models for all the news networks and generates headlines based on the fine-tuned models. No need to make any changes, "run all" should work.

## Other Notes
For the scraper, some of the networks require opening a chromium browser to find the html. Sometimes the bot protection kicks in chromium doesn't direct to the page with the headline html. When this happens just try running the scraper again later.

Fine-tuning the language model will create 9 separate fairly large folders in final_project directory. I would recommend deleting them. I made the mistake of trying to upload everything into my repo and it caused a mess because the folders were way too large. This is why when actually generating the headlines, I have the notebook use the fine-tuned models I already uploaded to huggingface rather than any local versions that are created.
