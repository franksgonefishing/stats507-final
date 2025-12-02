import lib.sentiment_analysis as sentiment_analysis
import lib.hard_coded_constants as news_sites
import calendar
import argparse
from datetime import date

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_type", type=str, nargs="+", required=True, choices=["pos_neg_neu", "emotion", "ALL"])
    parser.add_argument("--sites", type=str, nargs="+", required=True, choices=list(news_sites.NEWS_SITES_BASE_URL.keys()) + ["ALL"])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int)
    parser.add_argument("--ignore_sites", type=str, nargs="+", choices=list(news_sites.NEWS_SITES_BASE_URL.keys()))
    parser.add_argument("--use_generated_headlines", type=bool)

    args = parser.parse_args()

    today = date.today()

    current_year = today.year
    current_month = today.month
    current_day = today.day

    use_generated_headlines = args.use_generated_headlines

    analysis_type = args.analysis_type
    if analysis_type[0] == "ALL":
        analysis_type = ["pos_neg_neu", "emotion"]

    sites = args.sites
    if sites[0] == "ALL":
        sites = list(news_sites.NEWS_SITES_BASE_URL.keys())
    if args.ignore_sites:
        sites = [site for site in sites if site not in args.ignore_sites]

    year = args.year
    if year < 2001 or year > current_year:
        parser.error(f"Please enter a year between 2001 to {current_year} inclusive")

    month = args.month
    if month < 1 or month > 12:
        parser.error("Please enter a valid month between 1-12 inclusive")
    if year == current_year and month > current_month:
        parser.error(f"Error: {year}-{month} is in the future")

    if year == current_year and month == current_month:
        days_in_month = current_day
    else:
        days_in_month = calendar.monthrange(year, month)[1]

    if args.day:
        day = args.day
        if day < 1 or day > days_in_month:
            parser.error(f"Please enter a valid day for the month of {year}-{month}, between 1-{days_in_month} inclusive")
    else:
        day = None
    
    for analysis in analysis_type:
        if analysis == "pos_neg_neu":
            for site in sites:
                if site in ["Newsmax", "MSNBC"]:
                    sentiment_analysis.find_pos_neg_neu_sentiment(site, year, month, use_generated_headlines=use_generated_headlines)
                else:
                    if day:
                        sentiment_analysis.find_pos_neg_neu_sentiment(site, year, month, day, use_generated_headlines=use_generated_headlines)
                    else:
                        for i in range(1, days_in_month + 1):
                            sentiment_analysis.find_pos_neg_neu_sentiment(site, year, month, i, use_generated_headlines=use_generated_headlines)
        if analysis == "emotion":
            for site in sites:
                if site in ["Newsmax", "MSNBC"]:
                    sentiment_analysis.find_emotion_sentiment(site, year, month, use_generated_headlines=use_generated_headlines)
                else:
                    if day:
                        sentiment_analysis.find_emotion_sentiment(site, year, month, day, use_generated_headlines=use_generated_headlines)
                    else:
                        for i in range(1, days_in_month + 1):
                            sentiment_analysis.find_emotion_sentiment(site, year, month, i, use_generated_headlines=use_generated_headlines)