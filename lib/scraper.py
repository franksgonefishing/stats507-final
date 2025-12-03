from urllib.request import urlopen
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from lib.hard_coded_constants import NEWS_SITES_BASE_URL, INT_TO_MONTH_DICT, DATA_FILE_NAME, NEW_SITES_HTML_TAGS, BREITBART_CATEGORY_LIST

import os

import pandas as pd

# formatting the url for each news website
def format_request(news_site, year, month, day=None):
    if news_site == "Breitbart":
        # needs special processing
        extension = ""
    elif news_site == "Fox":
        extension = f"{year}/{INT_TO_MONTH_DICT[month]}/{day}"
    elif news_site == "MSNBC":
        extension = f"{year}/{INT_TO_MONTH_DICT[month]}"
    elif news_site == "Newsmax":
        extension = f"{year}/{month}"
    elif news_site == "Nypost":
        extension = f"{year}/{month:02d}/{day:02d}"
    elif news_site == "NYT":
        extension = f"{year}/{month:02d}/{day:02d}"
    elif news_site == "USAToday":
        extension = f"{year}/{INT_TO_MONTH_DICT[month]}/{day}/"
    elif news_site == "Washpost":
        extension = f"{year}/{month}/{day}"
    elif news_site == "WSJ":
        extension = f"{year}/{month:02d}/{day:02d}"
    else:
        extension = "broken"

    complete_url = NEWS_SITES_BASE_URL[news_site] + extension

    return complete_url


# function to parse html from news websites after it has been pulled from the website
def parse_html(html, news_site, year=None, month=None, day=None):
    soup = BeautifulSoup(html, "html.parser")
    
    headlines = []
    urls = []

    # html tags for each network is stored in hard_coded_constants
    for item in soup.select(NEW_SITES_HTML_TAGS[news_site]):
        headline = item.get_text(strip=True)
        if headline == "":
            continue
        url = item["href"]

        # Breitbart sometimes posts current articles even on these historical archive pages
        # avoiding the above issue by filtering articles to save by the date present in the url
        if news_site == "Breitbart":
            if f"/{year}/{month:02d}/{day:02d}/" in url:
                headlines.append(headline)
                urls.append(url)
            else:
                continue
        else:
            headlines.append(headline)
            urls.append(url)

    return headlines, urls


def write_headlines(news_site, year, month, day=None):

    request = format_request(news_site, year, month, day)

    # needs special processing
    # Breitbart historical archive is both by date and by a specific category
    if news_site == "Breitbart":
        headlines = []
        urls = []
        for b_cat in BREITBART_CATEGORY_LIST:
            b_request = request + f"{b_cat}/{year}/{month}/{day}"
            print(f"accessing: {b_request}")

            page = urlopen(b_request)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")

            add_headlines, add_urls = parse_html(html, news_site, year, month, day)
            headlines.extend(add_headlines)
            urls.extend(add_urls)

            print(f"{len(add_headlines)} headlines found at {b_request}")
    else:
        print(f"accessing: {request}")

        # special case, cannot access html via urlopen
        # have to open a chromium browser and extract html from there
        # cannot be a headless chromium browser either
        if news_site in ["Washpost", "WSJ"]:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False, args=["--window-size=600,300", "--window-position=1300,0"])
                context = browser.new_context(
                    viewport={"width": 600, "height": 300}
                )
                page = context.new_page()
                page.goto(request, wait_until="load", timeout=60000)
                html = page.content()
                context.close()
                browser.close()
                
        else:
            page = urlopen(request)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")

        headlines, urls = parse_html(html, news_site)

        print(f"{len(headlines)} headlines found at {request}")

    headlines_df = pd.DataFrame({
        "year": year,
        "month": month,
        "day": day,
        "network": news_site,
        "headline": headlines,
        "url": urls,
    })
    headlines_df = headlines_df.drop_duplicates()

    if os.path.isfile(DATA_FILE_NAME):
        # if csv exists we want to add to it
        existing_df = pd.read_csv(DATA_FILE_NAME, low_memory=False)

        # avoid adding duplicate headlines by checking network, date, and headline with existing entries
        mask = (existing_df["network"] == news_site) & (existing_df["year"] == year) & (existing_df["month"] == month)
        # add the optional condition only if day exists (MSNBC and Newsmax are only by month)
        if day:
            mask &= (existing_df["day"] == day)
        existing_headlines = existing_df[mask]["headline"].to_list()
        new_rows = headlines_df[~headlines_df["headline"].isin(existing_headlines)]

        # add headlines if there are headlines to add, otherwise print a message to the user
        if not new_rows.empty:
            new_rows = new_rows.reindex(columns=existing_df.columns)
            updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
            updated_df.to_csv(DATA_FILE_NAME, index=False)
            print(f"added {len(new_rows)} headlines to {DATA_FILE_NAME} for {news_site} on {year}-{month}{'-' + str(day) if day else ''}")
        else:
            print(f"no new headlines to add for {news_site} on {year}-{month}{'-' + str(day) if day else ''}")
    else:
        # csv doesn't exist, create new one
        headlines_df.to_csv(DATA_FILE_NAME, index=False)


# including function to delete headlines in case the csv gets 
# large enough where it is annoying to manually delete via excel
def delete_headlines(news_site, year, month, day=None):

    data_df = pd.read_csv(DATA_FILE_NAME, low_memory=False)

    mask = (data_df["network"] == news_site) & (data_df["year"] == year) & (data_df["month"] == month)
    if day:
        mask &= (data_df["day"] == day)
    num_rows = len(data_df[mask])
    print(f"dropping {num_rows} rows for {news_site} on {year}-{month}{'-' + str(day) if day else ''}")

    data_df = data_df[~mask]

    data_df.to_csv(DATA_FILE_NAME, index=False)