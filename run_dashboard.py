from dash import Dash, html, dcc, callback, Output, Input, State

from lib.hard_coded_constants import DATA_FILE_NAME, IGNORE_NGRAMS_FILE_NAME, NEWS_CATEGORIES, NEWS_SITES_BASE_URL, NEWS_SITE_COLORS, EMOTION_CATEGORIES, EMOTION_COLORS, GENERATED_HEADLINES_FILE_NAME
from lib.dashboard_functions import filter_df, error_fig, p_val_colors

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._param_validation import InvalidParameterError

from scipy.stats import ks_2samp, chi2_contingency

import prince

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# read csvs for both the normal heaadlines and the generated headlines
# add a column to identify which is which for filtering purposes
normal_headline_df = pd.read_csv(DATA_FILE_NAME)
normal_headline_df["source"] = "real_headlines"

generated_headline_df = pd.read_csv(GENERATED_HEADLINES_FILE_NAME)
generated_headline_df["network"] = generated_headline_df["network"] + "_generated"
generated_headline_df["source"] = "generated_headlines"

# create the big dataframe feeding into all the visualizations
df = pd.concat([normal_headline_df, generated_headline_df]).reset_index(drop=True)

# get the list of possible networks to filter on to use as filtering options
only_real_network_list = sorted(list(NEWS_SITES_BASE_URL.keys()))
generated_network_list = sorted(list(generated_headline_df["network"].unique()))

# zipping the possible networks to filter on in a specific order so 
# that the formatting looks better for the options in the dashboard
all_network_list_with_generated_headlines = []
for l, r in zip(only_real_network_list, generated_network_list):
    all_network_list_with_generated_headlines.append(l)
    all_network_list_with_generated_headlines.append(r)

# getting the min and max month for filtering option purposes
month_min = min(df["month"])
month_max = max(df["month"])

app = Dash(suppress_callback_exceptions=True)

# high level layout for the dashboard
app.layout = html.Div([
    # used to refresh the correspondence analysis graph
    # not visible in the dashboard
    dcc.Store(id="refresh-ca-store", data=0),
    # title
    html.Div(
        children="STATS 507: Final Project Dashboard - huafrank",
        style={
            "fontSize": "32px",
            "fontWeight": "bold",
            "color": "#000000",
            "textAlign": "center",
            "fontFamily": "Times New Roman"
        }
    ),
    html.Hr(),
    # tab selector at the top of the screen
    dcc.Tabs(
        id="tab-selector",
        value="ca-tab",  # default selected tab
        className="custom-tabs-container",
        children=[
            dcc.Tab(
                label="Correspondence Analysis",
                value="ca-tab",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Sentiment Analysis Density Comparison",
                value="sentiment-tab",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Emotion Analysis Proportion Comparison",
                value="emotion-tab",
                className="custom-tab",
                selected_className="custom-tab--selected"
            )
        ]
    ),
    # content visualized depending on the selected tab
    html.Div(id="tabs-content")
])

##########################################################################################
# BEGIN TAB DISPLAY FORMATTING
##########################################################################################

# adjusts the displayed content depending on the selected tab
LEFT_COL_WIDTH = "30%"
RIGHT_COL_WIDTH = "65%"
@callback(
    Output("tabs-content", "children"),
    Input("tab-selector", "value")
)
def render_tab_content(tab):
    # tab for correspondence analysis
    if tab == "ca-tab":
        return html.Div([
            html.Div(
                children=[
                    html.Br(),
                    # description at the top of the screen
                    "My analysis here was inspired by ",
                    html.A("this paper", href="https://arxiv.org/abs/2303.15708", target="_blank"),
                    " which focuses on broader patterns across many years, while my analysis is more focused on recent events.",
                    html.Br(),
                    html.Br(),
                    # description still at the top of the screen
                    html.H4("CA Description"),
                    html.P([
                        """
                        Correspondence Analysis is an analysis technique for categorical data that represents data
                        in a low-dimensional Euclidean space similar to how Principal Component Analysis is used to represent
                        continuous data in low-dimensional space. 
                        """,
                        html.A("Click here", href="https://en.wikipedia.org/wiki/Correspondence_analysis", target="_blank"),
                        """
                        for the Wiki article or 
                        """,
                        html.A("click here", href="https://www.geeksforgeeks.org/data-analysis/what-is-correspondence-analysis/", target="_blank"),
                        """
                        for a Geeksforgeeks article. 
                        """
                    ]),
                    html.P(
                        """
                        Red/Teal Dots: These represent individual news networks (teal dots represent the generated
                        versions). When these networks are closer to each other, the networks contain similar
                        ngram usage. When these networks are further form each other, these networks more often
                        use different ngrams and don't have as much overlap in ngram usage.
                        """
                    ),
                    html.P(
                        """
                        Blue Dots: These represent different ngrams. What matters is distance from the origin (represented
                        by the central X). The closer to the origin, the more these ngrams are used by each network. The
                        greater distance from the origin in the direction of a network, the more often that ngram is associated
                        with that news network.
                        """
                    ),
                    html.P(
                        """
                        Some examples of headlines to filter on include: 'trump', 'kirk', 'shutdown', 'mamdani',
                        'israel', 'china', 'russia', 'war', 'tariff', 'court', 'national guard', 'ukrai' 
                        (covers both 'ukraine' and 'ukrainian')
                        """
                    ),
                    html.Hr(),
                ]
            ),
            html.Br(),
            # split the screen into two sections, one on the left and one on the right
            # left section contains filter selections
            # right section contains the visualization
            html.Div(
                style={"display": "flex", "gap": "40px", "alignItems": "flex-start"},
                children=[
                    # LEFT COLUMN -------------------------------------------------------
                    html.Div(
                        style={"width": LEFT_COL_WIDTH},
                        children=[
                            # additional filters for extended exploration of data
                            html.Details(
                                children=[
                                    html.Summary("Click here for additional filtering options"),
                                    html.Label("Month range selector (everything is in 2025, must include Oct for generated headline comparison):"),
                                    html.Div(
                                        dcc.RangeSlider(
                                            month_min,
                                            month_max, 
                                            1, 
                                            value=[month_min, month_max], 
                                            id="months-range-selection",
                                            marks={
                                                7: "July",
                                                8: "Aug",
                                                9: "Sep",
                                                10: "Oct"
                                            },
                                        ),
                                    ),
                                    html.Label("Categories outside of politics are less stable due to high variance in headline subject matter"),
                                    dcc.Checklist(
                                        NEWS_CATEGORIES,
                                        ["politics"],
                                        id="checklist-selection"
                                    ),
                                    html.Label("See data/ignore_these_ngrams.csv for the list, some news companies reference themselves like Newsmax which skews distinct headline ngrams"),
                                    dcc.RadioItems(
                                        options=[
                                            {"label": "Exclude preselected ngrams", "value": True},
                                            {"label": "Include all ngrams", "value": False}
                                        ],
                                        value=True, 
                                        id="ignore-preselected-ngrams",
                                    ),
                                ]
                            ),
                            html.Br(),
                            # standard filters visible by default
                            dcc.RadioItems(
                                options=[
                                    {"label": "Include generated headlines", "value": True},
                                    {"label": "Only show real headlines", "value": False},
                                ],
                                value=False,
                                id="include-generated-headlines-ca",
                            ),
                            html.Br(),
                            dcc.RadioItems(
                                options=[
                                    {"label": "Show ngrams", "value": True},
                                    {"label": "Hide ngrams", "value": False},
                                ],
                                value=True,
                                id="show-ngrams-ca",
                            ),
                            html.Br(),
                            html.Label("Adjust the range of n in ngrams that the analysis is conducted on:"),
                            html.Div(
                                dcc.RangeSlider(
                                    2, 5, 1, value=[2, 4], id="n-gram-selection"
                                ),
                            ),
                            html.Br(),
                            html.Label("Min occurrence of ngram to be considered for the analysis: "),
                            dcc.Input(
                                id="n-gram-occurrence",
                                type="number",
                                value=10,
                                min=5,
                                style={"width": "10.5%"},
                            ),
                            html.Br(),
                            html.Label("[Optional] Filter headlines that contain: "),
                            dcc.Input(
                                id="n-gram-single-word-filter",
                                type="text",
                                placeholder="Enter a substring here",
                                style={"width": "39%"},
                            ),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            # additional tool for removing ngrams from the analysis
                            html.Div([
                                html.Div(dcc.Input(id="input-on-submit", type="text")),
                                html.Button("Submit", id="submit-ngram-for-ignore", n_clicks=0),
                                html.Div(id="container-button-basic",
                                        children="Enter an n-gram to ignore for future comparisons")
                            ])
                        ],
                    ),
                    # RIGHT COLUMN ------------------------------------------------------
                    html.Div(
                        style={
                            "width": RIGHT_COL_WIDTH, 
                            "padding": "0 10px",
                            "display": "flex",
                            "justifyContent": "center",   # horizontally center children
                        },
                        children=[
                            # visualization on the right of the screen
                            dcc.Loading(
                                id="loading-ca-graph",
                                type="default",
                                children=dcc.Graph(
                                    figure={}, 
                                    id="controls-and-graph", 
                                    style={"width": "750px", "height": "600px"},
                                    config={"responsive": False, "displayModeBar": False}
                                ),
                                delay_show=250,
                                delay_hide=250
                            ),
                        ]
                    ),
                ],
            ),
            
        ])
    # VADER sentiment tab
    elif tab == "sentiment-tab":
        return html.Div([
            # description at the top of the screen
            html.Div(
                children=[
                    html.Br(),
                    "The scores here come from ",
                    html.A("VADER Sentiment Analysis", href="https://pypi.org/project/vaderSentiment/", target="_blank"),
                    ", a lexicon and rule-based sentiment analysis tool.",
                    html.Br(),
                    html.Br(),
                ]
            ),
            html.Br(),
            html.Div(
                style={"display": "flex", "gap": "40px"},
                children=[
                    # LEFT COLUMN  ------------------------------------------------------
                    html.Div(
                        style={"width": LEFT_COL_WIDTH},
                        children=[
                            # default filter items
                            dcc.RadioItems(
                                options=[
                                    {"label": "Include generated headlines", "value": True},
                                    {"label": "Only show real headlines", "value": False}
                                ],
                                value=False, 
                                id="include-generated-headlines-sentiment",
                            ),
                            html.Br(),
                            dcc.Checklist(
                                id="network-selection-sentiment",
                                options=[],
                                value=[],
                                labelStyle={"display": "block"}
                            ),
                            html.Br(),
                            html.Label("[Optional] Filter headlines that contain: "),
                            dcc.Input(
                                id="head-filter-sentiment",
                                type="text",
                                placeholder="Enter a substring here",
                                style={"width": "39%"},
                            ),
                        ],
                    ),
                    # RIGHT COLUMN  -----------------------------------------------------
                    html.Div(
                        style={"width": RIGHT_COL_WIDTH, "padding": "0 10px"},
                        # description of the analysis
                        children=[
                            html.H4("Sentiment Analysis Description"),
                            html.P(
                                """
                                Each color line represents a probability density for a different news network on the
                                sentiment score (between -1: negative and 1: positive). The graph underneath it plots each
                                individual headline at the respective sentiment score for each news network. You can see what
                                each headline actually reads by hovering over it with your mouse. You can also filter by specific 
                                headline subsets based on substrings in the headlines.
                                """
                            ),
                            html.P(
                                """
                                The matrix represents pairwise Kolmogorov-Smirnov (KS) tests between each news network (and 
                                generated_news_network headlines if selected). The KS test determines if two sample distributions
                                come from the same distribution by looking at the maximum vertical distance between the two CDFs. 
                                The null hypothesis for this test is that the two sample distributions ultimately come from the 
                                same distribution, while the alternative hypothesis is that the two samples come from different 
                                distributions. I did not make any adjustments for multiple hypothesis testing.
                                """
                            ),
                            html.P(
                                """
                                Some examples of headlines to filter on include: 'trump', 'kirk', 'shutdown', 'mamdani',
                                'israel', 'china', 'russia', 'war', 'tariff', 'court', 'national guard', 'ukrai' 
                                (covers both 'ukraine' and 'ukrainian')
                                """
                            ),
                        ],
                    ),
                ]
            ),
            html.Br(),
            # CENTERED ----------------------------------------------------------
            # additional filtering options for further exploration
            # done with the two separated columns
            # everything is centered in the screen again
            html.Details(
                children=[
                    html.Summary("Click here for additional filtering options"),
                    html.Label("Month range selector (everything is in 2025, must include Oct for generated headline comparison):"),
                    html.Div(
                        dcc.RangeSlider(
                            month_min,
                            month_max, 
                            1, 
                            value=[month_min, month_max], 
                            id="months-range-selection-sentiment",
                            marks={
                                7: "July",
                                8: "Aug",
                                9: "Sep",
                                10: "Oct"
                            },
                        ),
                        style={
                            "width": "30%",
                        }
                    ),
                    html.Label("Categories outside of politics are less stable due to high variance in headline subject matter"),
                    dcc.Checklist(
                        NEWS_CATEGORIES,
                        ["politics"],
                        id="checklist-selection-sentiment"
                    ),
                ]
            ),
            # GRAPHS ---------------------------------------------------------
            # scatter and rug plot at the top
            dcc.Loading(
                id="loading-distribution-sentiment",
                type="default",
                children=dcc.Graph(id="sentiment-distribution-plot"),
                delay_show=250,
                delay_hide=250
            ),
            # pairwise KS test matrix
            dcc.Loading(
                id="loading-matrix-sentiment",
                type="default",
                children=dcc.Graph(id="sentiment-matrix"),
                delay_show=250,
                delay_hide=250
            )
        ])
    # Emotion Sentiment tab
    elif tab == "emotion-tab":
        return html.Div([
            # header at the top
            html.Div(
                children=[
                    html.Br(),
                    "I utilized ",
                    html.A("this huggingface model", href="https://huggingface.co/michellejieli/emotion_text_classifier", target="_blank"),
                    " to classify the emotions from each headline. The creator fine-tuned the model on transcripts from the Friends show!",
                    html.Br(),
                    html.Br(),
                ]
            ),
            html.Br(),
            html.Div(
                style={"display": "flex", "gap": "40px"},
                children=[
                    # LEFT COLUMN  ------------------------------------------------------
                    html.Div(
                        style={"width": LEFT_COL_WIDTH},
                        # default filter options
                        children=[
                            dcc.RadioItems(
                                options=[
                                    {"label": "Include generated headlines", "value": True},
                                    {"label": "Only show real headlines", "value": False}
                                ],
                                value=False, 
                                id="include-generated-headlines-emotion",
                            ),
                            html.Br(),
                            dcc.Checklist(
                                id="network-selection-emotion",
                                options=[],
                                value=[],
                                labelStyle={"display": "block"}
                            ),
                            html.Br(),

                            html.Label("[Optional] Filter headlines that contain: "),
                            dcc.Input(
                                id="head-filter-emotion",
                                type="text",
                                placeholder="Enter a substring here",
                                style={"width": "39%"},
                            ),
                        ],
                    ),
                    # RIGHT COLUMN  -----------------------------------------------------
                    html.Div(
                        style={"width": RIGHT_COL_WIDTH, "padding": "0 10px"},
                        children=[
                            # description of the analysis
                            html.H4("Emotion Analysis Description"),
                            html.P(
                                """
                                Each stacked bar chart indicates the proportion of headlines categorized by the 
                                primary emotion captured in each headline. The emotions are based on 
                                "6 Ekman emotions and a neutral class": anger, 
                                disgust, fear, joy, neutrality, sadness, and surprise. You can filter down to a specific 
                                subset of headlines as well by utilizing the headline substring filter.
                                """
                            ),
                            html.P(
                                """
                                The matrix utilizes pairwise chi-squared contingency tests from SciPy, comparing the 
                                distribution of each news network (and generated news network if selected) to see if the news networks 
                                come from the same or different distributions. The null hypothesis is that the two networks come from 
                                the same distribution, and the alternative hypothesis is that they come from different distributions. 
                                No adjustment is made for multiple hypothesis testing.
                                """
                            ),
                            html.P(
                                """
                                Some examples of headlines to filter on include: 'trump', 'kirk', 'shutdown', 'mamdani',
                                'israel', 'china', 'russia', 'war', 'tariff', 'court', 'national guard', 'ukrai' 
                                (covers both 'ukraine' and 'ukrainian')
                                """
                            ),
                        ],
                    ),
                ]
            ),
            html.Br(),
            # CENTERED ------------------------------------------------------------------
            html.Details(
                children=[
                    # additional filtering options for further exploration
                    # done with the two separated columns
                    # everything is centered in the screen again
                    html.Summary("Click here for additional filtering options"),
                    html.Label("Month range selector (everything is in 2025, must include Oct for generated headline comparison):"),
                    html.Div(
                        dcc.RangeSlider(
                            month_min,
                            month_max, 
                            1, 
                            value=[month_min, month_max], 
                            id="months-range-selection-emotion",
                            marks={
                                7: "July",
                                8: "Aug",
                                9: "Sep",
                                10: "Oct"
                            },
                        ),
                        style={
                            "width": "30%",
                        }
                    ),
                    html.Label("Categories outside of politics are less stable due to high variance in headline subject matter"),
                    dcc.Checklist(
                        NEWS_CATEGORIES,
                        ["politics"],
                        id="checklist-selection-emotion"
                    ),
                ]
            ),
            # GRAPHS ----------------------------------------------------------------
            # stacked bar chart
            dcc.Loading(
                id="loading-stacked-bar-emotion",
                type="default",
                children=dcc.Graph(id="emotion-stacked-bar"),
                delay_show=250,
                delay_hide=250
            ),
            # pairwise chi-sq test matrix
            dcc.Loading(
                id="loading-matrix-emotion",
                type="default",
                children=dcc.Graph(id="emotion-matrix"),
                delay_show=250,
                delay_hide=250
            ),
        ])
##########################################################################################
# END TAB DISPLAY FORMATTING
##########################################################################################


##########################################################################################
# BEGIN FILTER OPTION CALLBACKS
##########################################################################################

# dynamic function to allow filtering on generated networks/headlines only if the option is selected
# this one controls the VADER sentiment tab
@callback(
    Output("network-selection-sentiment", "style"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_sentiment_checklist_columns(show_generated):
    if show_generated:
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "columnGap": "20px",
            "width": "400px",
        }
    else:
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr",
            "columnGap": "20px",
        }

    return style


# dynamic function to allow filtering on generated networks/headlines only if the option is selected
# this one controls the emotion sentiment tab
@callback(
    Output("network-selection-emotion", "style"),
    Input("include-generated-headlines-emotion", "value")
)
def update_emotion_checklist_columns(show_generated):
    if show_generated:
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "columnGap": "20px",
            "width": "400px",
        }
    else:
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr",
            "columnGap": "20px",
        }
        
    return style


# dynamic function to allow filtering on generated networks/headlines only if the option is selected
# this one controls the VADER sentiment tab
@callback(
    Output("network-selection-sentiment", "options"),
    Output("network-selection-sentiment", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_sentiment_network_selector(include_generated_headlines):
    network_selection = {
        True : all_network_list_with_generated_headlines,
        False : only_real_network_list
    }

    return network_selection[include_generated_headlines], only_real_network_list


# dynamic function to allow filtering on generated networks/headlines only if the option is selected
# this one controls the emotion sentiment tab
@callback(
    Output("network-selection-emotion", "options"),
    Output("network-selection-emotion", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_emotion_network_selector(include_generated_headlines):
    network_selection = {
        True : all_network_list_with_generated_headlines,
        False : only_real_network_list
    }

    return network_selection[include_generated_headlines], only_real_network_list


# controls for the tool that helps remove ngrams from the analysis for the Correspondence Analysis tab
@callback(
    Output("refresh-ca-store", "data"),  # new output
    Output("container-button-basic", "children"),
    Output("input-on-submit", "value"),
    Input("submit-ngram-for-ignore", "n_clicks"),
    State("input-on-submit", "value"),
    State("refresh-ca-store", "data"),
    prevent_initial_call=True
)
def remove_inputted_ngram(_, removed_ngram, current_data):
    if removed_ngram == "":
        return current_data + 1, "", ""
    else:
        ngram_ignore_df = pd.read_csv(IGNORE_NGRAMS_FILE_NAME)
        new_ngram = pd.DataFrame({"ignored_ngrams": [removed_ngram]})
        ngram_ignore_df = pd.concat([ngram_ignore_df, new_ngram])
        ngram_ignore_df.to_csv(IGNORE_NGRAMS_FILE_NAME, index=False)

        return current_data + 1, f"added '{removed_ngram}' to {IGNORE_NGRAMS_FILE_NAME}", ""


##########################################################################################
# END FILTER OPTION CALLBACKS
##########################################################################################


##########################################################################################
# BEGIN CORRESPONDENCE ANALYSIS TAB VISUALIZATIONS
##########################################################################################


# creates the visualization for the Correspondence Analysis tab
@callback(
    Output("controls-and-graph", "figure"),
    Input("ignore-preselected-ngrams", "value"),
    Input("months-range-selection", "value"),
    Input("checklist-selection", "value"),
    Input("show-ngrams-ca", "value"),
    Input("n-gram-selection", "value"),
    Input("n-gram-occurrence", "value"),
    Input("n-gram-single-word-filter", "value"),
    Input("include-generated-headlines-ca", "value"),
    Input("refresh-ca-store", "data")
)
def update_graph(ignore_preselected_ngrams, month_range, col_chosen, show_ngrams, n_gram_value, n_gram_occurrence_filter, headline_filter, include_generated_headlines, _):
    
    # so that if the input is None when the user is changing the value, the value will be at least 5
    n_gram_occurrence_filter = n_gram_occurrence_filter or 5

    if ignore_preselected_ngrams:
        IGNORE_NGRAMS = np.array(pd.read_csv(IGNORE_NGRAMS_FILE_NAME)["ignored_ngrams"])
    else:
        IGNORE_NGRAMS = np.array([])

    filtered_df = filter_df(df, month_range=month_range, news_category_selection=col_chosen, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    # filters are too specific and there's no headlines
    if len(filtered_df) == 0:
        return error_fig("No data available")

    try:
        # get the ngrams from the headlines
        vectorizer = CountVectorizer(
            ngram_range=(n_gram_value[0], n_gram_value[1]),
            stop_words=list(ENGLISH_STOP_WORDS), 
            min_df=n_gram_occurrence_filter
        )
        # X is a matrix of headlines on one axis, ngrams on the other axis
        X = vectorizer.fit_transform(filtered_df["headline"])
    except ValueError:
        # sometimes there are headlines and ngrams but the results are all lower than min_df
        return error_fig("Too few n-grams to analyze")
    
    # remove ngrams that are in ignore_these_ngrams.csv
    # some ngrams are unique to networks because they are self-referential
    # or some standard label, like "fox news digital"
    # these ngrams skew uniqueness for each network so the analysis is better when removed
    feature_names = vectorizer.get_feature_names_out()
    filter_ngrams = np.isin(feature_names, IGNORE_NGRAMS, invert=True)
    X = X[:, filter_ngrams]
    feature_names = feature_names[filter_ngrams]

    # the same as groupby
    # groups ngram usage by network, original X matrix is ngrams by headline
    networks = filtered_df["network"].to_numpy().reshape(len(filtered_df["network"]), 1)
    enc = OneHotEncoder(sparse_output=True)  # keep it sparse for speed
    G = enc.fit_transform(networks)
    CT = X.T @ G 

    network_names = enc.categories_[0]

    col_sums = np.asarray(CT.sum(axis=0)).flatten()
    nonzero_cols = col_sums > 0

    # there could be too few n-grams post filtering of n-grams above
    if nonzero_cols.sum() == 0:
        return error_fig("Too few n-grams to analyze")

    # removes networks that end up having 0 ngrams across the board
    CT = CT[:, nonzero_cols]
    network_names = network_names[nonzero_cols]

    n_total = CT.sum()

    # relative frequencies, full contingency table
    P = CT / n_total
    P = pd.DataFrame(P.toarray())

    try:
        ca = prince.CA(n_components=2)
        ca = ca.fit(P)

        row_coords = ca.row_coordinates(P)
        col_coords = ca.column_coordinates(P)
    except:
        # sometimes an error can pop up if there's only one network remaining even when the earlier checks don't pop an error
        return error_fig("Too few n-grams to analyze")

    # will anchor this to always be in the top left quadrant
    # keeps the graph stabalized
    preferred_ref = "NYT"

    # if "NYT" not in the data, fallback to first available network
    if preferred_ref in network_names:
        network_ref = preferred_ref
    else:
        network_ref = network_names[0]

    ref_idx = np.where(network_names == network_ref)[0][0]

    # get the reference coordinates
    ref_x = col_coords.iloc[ref_idx, 0]
    ref_y = col_coords.iloc[ref_idx, 1]

    # flip x-axis if ref_x > 0
    if ref_x > 0:
        col_coords.iloc[:, 0] *= -1
        row_coords.iloc[:, 0] *= -1

    # flip y-axis if ref_y < 0
    if ref_y < 0:
        col_coords.iloc[:, 1] *= -1
        row_coords.iloc[:, 1] *= -1

    # scale ngram points so that no ngram point is further than the furthest network point
    # helps with the graph visualization
    # compute max distance from origin for column and row coordinates
    col_max = np.max(np.sqrt(col_coords.iloc[:, 0] ** 2 + col_coords.iloc[:, 1] ** 2))
    row_max = np.max(np.sqrt(row_coords.iloc[:, 0] ** 2 + row_coords.iloc[:, 1] ** 2))
    scale_factor = 0.8 * col_max / row_max
    row_coords_scaled = row_coords.copy()
    row_coords_scaled.iloc[:, 0] *= scale_factor
    row_coords_scaled.iloc[:, 1] *= scale_factor

    # create separate colored points for generated networks vs base networks
    # Define colors for base networks vs generated networks
    base_color = "#E41A1C"
    generated_color = "#377EB8"
    # Boolean mask for which points are generated
    is_generated = np.array([("_generated" in name) for name in network_names])
    # Split the data
    x_gen = col_coords.iloc[:, 0][is_generated]
    y_gen = col_coords.iloc[:, 1][is_generated]
    text_gen = [name for name in network_names if "_generated" in name]
    x_base = col_coords.iloc[:, 0][~is_generated]
    y_base = col_coords.iloc[:, 1][~is_generated]
    text_base = [name for name in network_names if "_generated" not in name]

    fig = go.Figure()
    # --- TRACE 1: Generated ---
    fig.add_trace(go.Scatter(
        x=x_gen,
        y=y_gen,
        mode="markers+text",
        text=text_gen,
        hoverinfo="text",
        textposition="top center",
        marker=dict(size=10, color=generated_color),
        name='generated_headlines',
    ))

    # --- TRACE 2: Base (Non-generated) ---
    fig.add_trace(go.Scatter(
        x=x_base,
        y=y_base,
        mode="markers+text",
        text=text_base,
        hoverinfo="text",
        textposition="top center",
        marker=dict(size=10, color=base_color),
        name="headlines",
    ))

    if show_ngrams:
        # Scatter for n-grams
        fig.add_trace(go.Scatter(
            x=row_coords_scaled.iloc[:,0],
            y=row_coords_scaled.iloc[:,1],
            mode="markers",
            text=feature_names,
            hoverinfo="text",
            marker=dict(size=4, color='blue', opacity=0.2),
            name="n-grams"
        ))
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="black",
                    line=dict(width=0.5)
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.update_layout(
        legend=dict(
            x=1,
            y=1,
            orientation="h",
            yanchor="bottom",
            xanchor="right"
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks="",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks="",
        ),
        margin=dict(l=50, r=50, t=0, b=0),
        width=800,
        height=600
    )
    fig.update_traces(cliponaxis=False)

    return fig


##########################################################################################
# END CORRESPONDENCE ANALYSIS TAB VISUALIZATIONS
##########################################################################################


##########################################################################################
# BEGIN SENTIMENT ANALYSIS TAB VISUALIZATIONS
##########################################################################################


# creates the density plot and rug headline viewer for the VADER tab
@callback(
    Output("sentiment-distribution-plot", "figure"),
    Input("months-range-selection-sentiment", "value"),
    Input("checklist-selection-sentiment", "value"),
    Input("network-selection-sentiment", "value"),
    Input("head-filter-sentiment", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_sentiment_graph(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    # filters are too specific and there's no headlines
    if len(filtered_df) == 0:
        return error_fig("No data available")

    networks = filtered_df["network"].unique()
    data = [filtered_df[filtered_df["network"] == network]["vader_compound_score"] for network in networks]
    colors = [NEWS_SITE_COLORS[network] for network in networks]

    try:
        fig = ff.create_distplot(data, networks, show_hist=False, colors=colors)
    except:
        # error pops up when filtering too specifically on a certain headline
        return error_fig("too few headlines to form a distribution (a network might only have 1 headline)")

    fig.update_layout(title="Sentiment Density Distribution Plot")
    fig.update_yaxes(visible=False)
    # Add custom hover text to rug traces
    # Rug traces are usually the last 'len(networks)' traces in the figure
    for i, network in enumerate(networks):
        rug_trace_index = -len(networks) + i  # Get correct trace
        rug_headlines = filtered_df[filtered_df["network"] == network]["headline"]

        fig.data[rug_trace_index].text = rug_headlines
        fig.data[rug_trace_index].hovertemplate = "%{text}<br>Score: %{x}<extra></extra>"

    return fig


# creates the pairwise KS test matrix for the VADER sentiment tab
@callback(
    Output("sentiment-matrix", "figure"),
    Input("months-range-selection-sentiment", "value"),
    Input("checklist-selection-sentiment", "value"),
    Input("network-selection-sentiment", "value"),
    Input("head-filter-sentiment", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_sentiment_matrix(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)
    
    # filters are too specific and there's no headlines
    if len(filtered_df) == 0:
        return error_fig("No data available")
    
    networks = filtered_df["network"].unique()

    pval_matrix = pd.DataFrame(index=networks, columns=networks)

    for net1 in networks:
        for net2 in networks:
            scores1 = filtered_df[filtered_df["network"] == net1]["vader_compound_score"]
            scores2 = filtered_df[filtered_df["network"] == net2]["vader_compound_score"]
            
            if net1 == net2:
                pval_matrix.loc[net1, net2] = np.nan 
            else:
                _, pval = ks_2samp(scores1, scores2)
                pval_matrix.loc[net1, net2] = pval

    fig = px.imshow(
        pval_matrix,
        text_auto=".3f",
        color_continuous_scale=p_val_colors(0.05),
        zmin=0,
        zmax=1,
        labels=dict(color="p-value"),
        title="Pairwise Sentiment Distribution Comparison (KS p-values)"
    )
    
    return fig


##########################################################################################
# END SENTIMENT ANALYSIS TAB VISUALIZATIONS
##########################################################################################


##########################################################################################
# BEGIN EMOTION ANALYSIS TAB VISUALIZATIONS
##########################################################################################


# creates the stacked bar chart for the emotions tab
@callback(
    Output("emotion-stacked-bar", "figure"),
    Input("months-range-selection-emotion", "value"),
    Input("checklist-selection-emotion", "value"),
    Input("network-selection-emotion", "value"),
    Input("head-filter-emotion", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_emotion_bar(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    # filters are too specific and there's no headlines
    if len(filtered_df) == 0:
        return error_fig("No data available")
    
    # find % of emotion per network
    emotion_df = pd.crosstab(filtered_df["network"], filtered_df["primary_emotion"])
    emotion_df = emotion_df.div(emotion_df.sum(axis=1), axis=0).reset_index()
    emotion_df = emotion_df.melt(
        id_vars="network", 
        var_name="primary_emotion", 
        value_name="percentage"
    )
    emotion_df["percentage"] = emotion_df["percentage"] * 100

    # create the graph
    fig = px.bar(
        emotion_df,
        x="network",
        y="percentage",
        color="primary_emotion",
        text="percentage",
        title="Percentage of Each Emotion by Network",
        labels={
            "percentage": "Percent",
            "network": "Network", 
            "primary_emotion": "Emotion"
        },
        color_discrete_map=EMOTION_COLORS
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        hovertemplate="Emotion: %{fullData.name}<br>"
                      "Network: %{x}<br>"
                      "Percentage: %{y:.1f}<br>"
                      "<extra></extra>",
        cliponaxis=False
    )
    fig.update_layout(barmode="stack")

    return fig


# creates the pairwise chi-sq test matrix
@callback(
    Output("emotion-matrix", "figure"),
    Input("months-range-selection-emotion", "value"),
    Input("checklist-selection-emotion", "value"),
    Input("network-selection-emotion", "value"),
    Input("head-filter-emotion", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_emotion_matrix(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    # filters are too specific and there's no headlines
    if len(filtered_df) == 0:
        return error_fig("No data available")

    networks = filtered_df["network"].unique()

    pval_matrix = pd.DataFrame(index=networks, columns=networks)

    for net1 in networks:
        for net2 in networks:
            if net1 == net2:
                pval_matrix.loc[net1, net2] = np.nan
                continue
            
            # Count primary emotions for each network
            counts1 = filtered_df[filtered_df["network"] == net1]["primary_emotion"].value_counts()
            counts2 = filtered_df[filtered_df["network"] == net2]["primary_emotion"].value_counts()
            
            # Align with full emotion category list
            all_emotions = EMOTION_CATEGORIES
            counts1 = counts1.reindex(all_emotions, fill_value=0)
            counts2 = counts2.reindex(all_emotions, fill_value=0)

            # If only one category left, p-value = 1.0
            if len(counts1) < 2:
                pval_matrix.loc[net1, net2] = 1.0
                continue

            contingency = pd.DataFrame([counts1, counts2])

            # Haldane–Anscombe correction
            if (contingency.values == 0).any():
                contingency = contingency + 0.5

            # Run chi-square
            _, p, _, _ = chi2_contingency(contingency)
            pval_matrix.loc[net1, net2] = p

    # Create heatmap
    fig = px.imshow(
        pval_matrix,
        text_auto=".3f",
        color_continuous_scale=p_val_colors(0.05),
        zmin=0,
        zmax=1,
        labels=dict(color="p-value"),
        title="Pairwise Emotion Distribution Comparison (Chi-squared p-values)"
    )

    return fig


##########################################################################################
# END EMOTION ANALYSIS TAB VISUALIZATIONS
##########################################################################################


if __name__ == "__main__":
    app.run(debug=False)