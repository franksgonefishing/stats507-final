NEWS_CATEGORIES = ["politics", "sports", "business", "entertainment", "technology", "health"]

BREITBART_CATEGORY_LIST = [
    "politics",
    "entertainment",
    "the-media",
    "economy",
    "europe",
    "border",
    "middle-east",
    "africa",
    "asia",
    "latin-america",
    "sports"
]

GENERATED_HEADLINES_FILE_NAME = "data/generated_headlines_data.csv"

DATA_FILE_NAME = "data/headlines_data.csv"

IGNORE_NGRAMS_FILE_NAME = "data/ignore_these_ngrams.csv"

ADDITIONAL_STOP_WORDS = []

NEWS_CATEGORY_CLASSIFICATION_MODEL = "facebook/bart-large-mnli"

EMOTION_ANALYSIS_CLASSIFICATION_MODEL = "michellejieli/emotion_text_classifier"

BASE_LANGUAGE_MODEL = "EleutherAI/pythia-70m-deduped"

EMOTION_CATEGORIES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

NEWS_SITES_BASE_URL = {
    "Breitbart": "https://www.breitbart.com/",
    "Fox": "https://www.foxnews.com/html-sitemap/",
    "MSNBC": "https://www.ms.now/archives/articles/",
    "Newsmax": "https://www.newsmax.com/archives/all/",
    "Nypost": "https://nypost.com/",
    "NYT": "https://www.nytimes.com/sitemap/",
    "USAToday": "https://www.usatoday.com/sitemap/",
    "Washpost": "https://www.washingtonpost.com/sitemap/",
    "WSJ": "https://www.wsj.com/news/archive/",
}

NEW_SITES_HTML_TAGS = {
    "Breitbart": "div.tC h2 a",
    "Fox": "li.article-item a.sitemap-grid-link",
    "MSNBC": "p.post-title a",
    "Newsmax": "li.archiveRepeaterLI h5.archiveH5 a",
    "Nypost": "div.story__text h3.story__headline a",
    "NYT": "ul.css-d7lzgg li a",
    "USAToday": "a.gnt_sm_a.gnt_sm_a_li",
    "Washpost": "ul[data-testid='sitemap-stories'] li a",
    "WSJ": "h3.css-fsvegl a"
}

NEWS_SITE_COLORS = {
    # --- BREITBART (magenta family) ---
    "Breitbart": "#D81B60",            # strong raspberry
    "Breitbart_generated": "#F48FB1",  # light pink tint

    # --- FOX (blue family) ---
    "Fox": "#1E88E5",                  # vivid blue
    "Fox_generated": "#90CAF9",        # light sky-blue

    # --- MSNBC (orange family) ---
    "MSNBC": "#FB8C00",                # bright orange
    "MSNBC_generated": "#FFCC80",      # soft pastel orange

    # --- NEWSMAX (green family) ---
    "Newsmax": "#43A047",              # medium green
    "Newsmax_generated": "#A5D6A7",    # mint green

    # --- NYPOST (red family) ---
    "Nypost": "#E53935",               # strong red
    "Nypost_generated": "#EF9A9A",     # salmon tint

    # --- NYT (purple family) ---
    "NYT": "#8E24AA",                  # rich purple
    "NYT_generated": "#CE93D8",        # lavender tint

    # --- USA TODAY (yellow/olive family) ---
    "USAToday": "#FDD835",             # warm yellow
    "USAToday_generated": "#FFF176",   # soft pale yellow

    # --- WASHINGTON POST (brown family) ---
    "Washpost": "#6D4C41",             # deep brown
    "Washpost_generated": "#BCAAA4",   # light beige-brown

    # --- WALL STREET JOURNAL (teal family) ---
    "WSJ": "#00897B",                  # teal green
    "WSJ_generated": "#80CBC4",        # light teal tint
}

INT_TO_MONTH_DICT = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december"
}