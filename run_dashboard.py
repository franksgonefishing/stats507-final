from dash import Dash, html, dcc, callback, Output, Input, State

from lib.hard_coded_constants import DATA_FILE_NAME, IGNORE_NGRAMS_FILE_NAME, NEWS_CATEGORIES, ADDITIONAL_STOP_WORDS, NEWS_SITES_BASE_URL, NEWS_SITE_COLORS, EMOTION_CATEGORIES, GENERATED_HEADLINES_FILE_NAME
from lib.dashboard_functions import filter_df, error_fig

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import text 
from sklearn.utils._param_validation import InvalidParameterError

from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency, fisher_exact

import prince
import warnings

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np

normal_headline_df = pd.read_csv(DATA_FILE_NAME)
normal_headline_df["source"] = "real_headlines"

generated_headline_df = pd.read_csv(GENERATED_HEADLINES_FILE_NAME)
generated_headline_df["network"] = generated_headline_df["network"] + "_generated"
generated_headline_df["source"] = "generated_headlines"

df = pd.concat([normal_headline_df, generated_headline_df]).reset_index(drop=True)

only_real_network_list = sorted(list(NEWS_SITES_BASE_URL.keys()))
generated_network_list = sorted(list(generated_headline_df["network"].unique()))

all_network_list_with_generated_headlines = []
for l, r in zip(only_real_network_list, generated_network_list):
    all_network_list_with_generated_headlines.append(l)
    all_network_list_with_generated_headlines.append(r)

month_min = min(df["month"])
month_max = max(df["month"])

app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div([
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

    html.Div(id="tabs-content")
])


LEFT_COL_WIDTH = "30%"
RIGHT_COL_WIDTH = "65%"
@callback(
    Output("tabs-content", "children"),
    Input("tab-selector", "value")
)
def render_tab_content(tab):
    if tab == "ca-tab":
        # Custom inputs and graph for Tab 1
        return html.Div([
            html.Div(
                children=[
                    html.Br(),
                    "My analysis here was inspired by ",
                    html.A("this paper", href="https://arxiv.org/abs/2303.15708", target="_blank"),
                    " which focuses on broader patterns across many years, while my analysis is more focused on recent events.",
                    html.Br(),
                    html.Br(),
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
            html.Div(
                style={"display": "flex", "gap": "40px", "alignItems": "flex-start"},
                children=[
                    # LEFT COLUMN -------------------------------------------------------
                    html.Div(
                        style={"width": LEFT_COL_WIDTH},
                        children=[
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
                            html.Div([
                                html.Div(dcc.Input(id='input-on-submit', type='text')),
                                html.Button('Submit', id='submit-ngram-for-ignore', n_clicks=0),
                                html.Div(id='container-button-basic',
                                        children='Enter an n-gram to ignore for future comparisons')
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
                            dcc.Loading(
                                id="loading-graph",
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
    elif tab == "sentiment-tab":
        # Custom inputs and graph for Tab 2
        return html.Div([
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
                                id="network-selection-tab2",
                                options=[],
                                value=[],
                                labelStyle={"display": "block"}
                            ),
                            html.Br(),
                            html.Label("[Optional] Filter headlines that contain: "),
                            dcc.Input(
                                id="head-filter-tab2",
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
                            id="months-range-selection-tab2",
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
                        id="checklist-selection-tab2"
                    ),
                ]
            ),
            dcc.Loading(
                id="loading-graph-2",
                type="default",
                children=dcc.Graph(id='graph-tab-2'),
                delay_show=250,
                delay_hide=250
            ),
            dcc.Loading(
                id="loading-matrix-2",
                type="default",
                children=dcc.Graph(id='matrix-tab-2'),
                delay_show=250,
                delay_hide=250
            )
        ])
    elif tab == "emotion-tab":
        # Custom inputs and graph for Tab 3
        return html.Div([
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
                                id="network-selection-tab3",
                                options=[],
                                value=[],
                                labelStyle={"display": "block"}
                            ),
                            html.Br(),

                            html.Label("[Optional] Filter headlines that contain: "),
                            dcc.Input(
                                id="head-filter-tab3",
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
                            id="months-range-selection-tab3",
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
                        id="checklist-selection-tab3"
                    ),
                ]
            ),
            dcc.Loading(
                id="loading-chart-3",
                type="default",
                children=dcc.Graph(id='chart-tab-3'),
                delay_show=250,
                delay_hide=250
            ),
            dcc.Loading(
                id="loading-graph-3",
                type="default",
                children=dcc.Graph(id='hist-tab-3'),
                delay_show=250,
                delay_hide=250
            ),
        ])


@callback(
    Output('chart-tab-3', 'figure'),
    Input('months-range-selection-tab3', 'value'),
    Input("checklist-selection-tab3", "value"),
    Input("network-selection-tab3", "value"),
    Input("head-filter-tab3", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_tab3_hist(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)
    # Calculate counts per network × emotion
    emotion_counts = pd.crosstab(filtered_df['network'], filtered_df['primary_emotion'])

    # Convert counts to percentages per network
    emotion_percent = emotion_counts.div(emotion_counts.sum(axis=1), axis=0).reset_index()

    # Convert to long format for Plotly
    emotion_long = emotion_percent.melt(id_vars='network', 
                                        var_name='primary_emotion', 
                                        value_name='percentage')
    
    # Multiply by 100 for display
    emotion_long['percentage_display'] = emotion_long['percentage'] * 100

    fig = px.bar(
        emotion_long,
        x='network',
        y='percentage',
        color='primary_emotion',
        text='percentage_display',  # Show as 0–100
        title='Percentage of Each Emotion by Network',
        labels={'percentage': 'Percentage', 'network': 'Network', 'primary_emotion': 'Emotion'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Optional: format y-axis as percentage
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(texttemplate='%{text:.1f}%')
    fig.update_layout(barmode='stack')

    return fig


@callback(
    Output('hist-tab-3', 'figure'),
    Input('months-range-selection-tab3', 'value'),
    Input("checklist-selection-tab3", "value"),
    Input("network-selection-tab3", "value"),
    Input("head-filter-tab3", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_tab3_hist(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    # Example: contingency table of counts
    contingency = pd.crosstab(filtered_df['network'], filtered_df['primary_emotion'])

    networks = filtered_df['network'].unique()
    n = len(networks)

    pval_matrix = pd.DataFrame(index=networks, columns=networks)

    for i, net1 in enumerate(networks):
        for j, net2 in enumerate(networks):
            if i == j:
                pval_matrix.loc[net1, net2] = np.nan
                continue
            
            # Count primary emotions for each network
            counts1 = filtered_df[filtered_df['network']==net1]['primary_emotion'].value_counts()
            counts2 = filtered_df[filtered_df['network']==net2]['primary_emotion'].value_counts()
            
            # Align with full emotion category list
            all_emotions = EMOTION_CATEGORIES
            counts1 = counts1.reindex(all_emotions, fill_value=0)
            counts2 = counts2.reindex(all_emotions, fill_value=0)

            # Remove categories where BOTH are zero (no information)
            mask = ~((counts1 == 0) & (counts2 == 0))
            counts1 = counts1[mask]
            counts2 = counts2[mask]

            # If only one category left → completely identical, p-value = 1.0
            if len(counts1) < 2:
                pval_matrix.loc[net1, net2] = 1.0
                continue

            contingency = pd.DataFrame([counts1, counts2])

            # ---------------------------------------------------------
            # ⭐ Apply pseudocount 0.5 if any zeros exist
            #    (prevents zero expected frequencies)
            # ---------------------------------------------------------
            if (contingency.values == 0).any():
                contingency = contingency + 0.5

            # Run chi-square
            chi2, p, dof, expected = chi2_contingency(contingency)

            pval_matrix.loc[net1, net2] = p

    # Define colors
    light_blue = "rgb(173,216,230)"   # for p < 0.05
    light_red  = "rgb(255,182,193)"   # for p ≥ 0.05

    # Compute normalized breakpoint for the colorscale
    p_min = float(np.min(pval_matrix))
    p_max = float(np.max(pval_matrix))
    threshold = 0.05

    # Avoid division by zero
    if p_max == p_min:
        p_max = p_min + 1e-9

    t = (threshold - p_min) / (p_max - p_min)
    t = max(0.0, min(1.0, t))   # clamp to [0,1]

    # Custom two-color colorscale with a sharp break
    colorscale = [
        [0.0, light_blue],
        [t,   light_blue],
        [t,   light_red],
        [1.0, light_red]
    ]

    # Create heatmap
    fig = px.imshow(
        pval_matrix,
        text_auto=".3f",
        color_continuous_scale=colorscale,
        zmin=p_min,
        zmax=p_max,
        labels=dict(x="Network 2", y="Network 1", color="p-value"),
        title="Pairwise Emotion Distribution Comparison (Chi-squared p-values)"
    )

    return fig


@callback(
    Output("network-selection-tab2", "style"),
    Input("include-generated-headlines-sentiment", "value")  # or whatever Input controls it
)
def update_checklist_columns(show_generated):
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


@callback(
    Output("network-selection-tab3", "style"),
    Input("include-generated-headlines-emotion", "value")  # or whatever Input controls it
)
def update_checklist_columns(show_generated):
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

@callback(
    Output("network-selection-tab2", "options"),
    Output("network-selection-tab2", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_tab2_network_selector(include_generated_headlines):
    network_selection = {
        True : all_network_list_with_generated_headlines,
        False : only_real_network_list
    }
    if include_generated_headlines:
        num_columns_to_show = 2
    else:
        num_columns_to_show = 1

    return network_selection[include_generated_headlines], only_real_network_list


@callback(
    Output("network-selection-tab3", "options"),
    Output("network-selection-tab3", "value"),
    Input("include-generated-headlines-emotion", "value")
)
def update_tab2_network_selector(include_generated_headlines):
    network_selection = {
        True : all_network_list_with_generated_headlines,
        False : only_real_network_list
    }

    return network_selection[include_generated_headlines], only_real_network_list


@callback(
    Output('matrix-tab-2', 'figure'),
    Input('months-range-selection-tab2', 'value'),
    Input("checklist-selection-tab2", "value"),
    Input("network-selection-tab2", "value"),
    Input("head-filter-tab2", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_tab2_matrix(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)
    networks = filtered_df['network'].unique()
    n = len(networks)

    # Initialize empty DataFrame for the p-values
    pval_matrix = pd.DataFrame(np.zeros((n, n)), index=networks, columns=networks)

    for i, net1 in enumerate(networks):
        for j, net2 in enumerate(networks):
            scores1 = filtered_df[filtered_df['network'] == net1]['vader_compound_score']
            scores2 = filtered_df[filtered_df['network'] == net2]['vader_compound_score']
            
            if i == j:
                pval_matrix.loc[net1, net2] = np.nan  # Optional: self-comparison is NaN
            else:
                stat, pval = ks_2samp(scores1, scores2)
                pval_matrix.loc[net1, net2] = pval
    
    # Define colors
    light_blue = "rgb(173,216,230)"   # < 0.05
    light_red  = "rgb(255,182,193)"   # >= 0.05

    # Compute normalized breakpoint for the colorscale
    p_min = float(np.min(pval_matrix))
    p_max = float(np.max(pval_matrix))
    threshold = 0.05

    # Avoid division issues
    if p_max == p_min:
        p_max = p_min + 1e-9

    t = (threshold - p_min) / (p_max - p_min)
    t = max(0.0, min(1.0, t))  # clamp

    # Two-color colorscale with a sharp break
    colorscale = [
        [0.0, light_blue],
        [t,   light_blue],
        [t,   light_red],
        [1.0, light_red]
    ]

    fig = px.imshow(
        pval_matrix,
        text_auto=".3f",
        color_continuous_scale=colorscale,
        zmin=p_min,
        zmax=p_max,
        labels=dict(x="Network 2", y="Network 1", color="p-value")
    )
    
    return fig


@callback(
    Output('graph-tab-2', 'figure'),
    Input('months-range-selection-tab2', 'value'),
    Input("checklist-selection-tab2", "value"),
    Input("network-selection-tab2", "value"),
    Input("head-filter-tab2", "value"),
    Input("include-generated-headlines-sentiment", "value")
)
def update_tab2_graph(month_range, news_category_selection, network_selection, headline_filter, include_generated_headlines):
    
    filtered_df = filter_df(df, month_range=month_range, news_category_selection=news_category_selection, network_selection=network_selection, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    networks = filtered_df["network"].unique()
    data = [filtered_df[filtered_df["network"] == network]["vader_compound_score"] for network in networks]
    colors = [NEWS_SITE_COLORS[network] for network in networks]

    fig = ff.create_distplot(data, networks, show_hist=False, colors=colors)
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


@callback(
    Output('container-button-basic', 'children'),
    Output("input-on-submit", "value"),
    Input('submit-ngram-for-ignore', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, removed_ngram):
    ngram_ignore_df = pd.read_csv(IGNORE_NGRAMS_FILE_NAME)
    new_ngram = pd.DataFrame({'ignored_ngrams': [removed_ngram]})
    ngram_ignore_df = pd.concat([ngram_ignore_df, new_ngram])
    ngram_ignore_df.to_csv(IGNORE_NGRAMS_FILE_NAME, index=False)

    return f"added '{removed_ngram}' to {IGNORE_NGRAMS_FILE_NAME}", ""


@callback(
    Output(component_id="controls-and-graph", component_property="figure"),
    Input(component_id="ignore-preselected-ngrams", component_property="value"),
    Input(component_id="months-range-selection", component_property="value"),
    Input(component_id="checklist-selection", component_property="value"),
    Input(component_id="show-ngrams-ca", component_property="value"),
    Input(component_id="n-gram-selection", component_property="value"),
    Input(component_id="n-gram-occurrence", component_property="value"),
    Input(component_id="n-gram-single-word-filter", component_property="value"),
    Input(component_id="include-generated-headlines-ca", component_property="value"),
    Input(component_id="submit-ngram-for-ignore", component_property="n_clicks")
)
def update_graph(ignore_preselected_ngrams, month_range, col_chosen, show_ngrams, n_gram_value, n_gram_occurrence_filter, headline_filter, include_generated_headlines, filler):
    stop_words = list(text.ENGLISH_STOP_WORDS.union(ADDITIONAL_STOP_WORDS))
    n_gram_occurrence_filter = n_gram_occurrence_filter or 5
    if ignore_preselected_ngrams:
        IGNORE_NGRAMS = np.array(pd.read_csv(IGNORE_NGRAMS_FILE_NAME)["ignored_ngrams"])
    else:
        IGNORE_NGRAMS = np.array([])

    filtered_df = filter_df(df, month_range=month_range, news_category_selection=col_chosen, headline_contains=headline_filter, include_generated_headlines=include_generated_headlines)

    if len(filtered_df) == 0:
        return error_fig("No data available")

    try:
        vectorizer = CountVectorizer(
            ngram_range=(n_gram_value[0], n_gram_value[1]),
            stop_words=stop_words, 
            min_df=n_gram_occurrence_filter
        )
        X = vectorizer.fit_transform(filtered_df["headline"])
    except ValueError:
        return error_fig("Too few headlines available for analysis")
    feature_names = vectorizer.get_feature_names_out()

    mask = np.isin(feature_names, IGNORE_NGRAMS, invert=True)
    X = X[:, mask]
    feature_names = feature_names[mask]

    networks = filtered_df["network"].to_numpy().reshape(len(filtered_df["network"]), 1)
    enc = OneHotEncoder(sparse_output=True)  # keep it sparse
    G = enc.fit_transform(networks)          # shape: (n_docs × n_networks)
    CT = X.T @ G 

    network_names = enc.categories_[0]

    col_sums = np.asarray(CT.sum(axis=0)).flatten()
    nonzero_cols = col_sums > 0

    if nonzero_cols.sum() == 0:
        return error_fig("No networks with data available")

    # Apply mask
    CT = CT[:, nonzero_cols]                  # keep only usable networks
    network_names = network_names[nonzero_cols]
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # OPTIONAL: Remove n-grams (rows) that appear nowhere after filtering
    # (Sometimes min_df or ignoring removes everything for that term)
    # ---------------------------------------------------------------------------
    row_sums = np.asarray(CT.sum(axis=1)).flatten()
    nonzero_rows = row_sums > 0

    if nonzero_rows.sum() == 0:
        return error_fig("No n-grams satisfy your filters")

    CT = CT[nonzero_rows, :]
    feature_names = feature_names[nonzero_rows]

    n_total = CT.sum()

    # Step 1: relative frequencies
    P = CT / n_total
    P = pd.DataFrame(P.toarray())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ca = prince.CA(n_components=2)
            ca = ca.fit(P)

            row_coords = ca.row_coordinates(P)
            col_coords = ca.column_coordinates(P)
    except InvalidParameterError:
        error_fig("Too few headlines available for analysis")

    # Preferred reference network
    preferred_ref = "NYT"

    # Choose NYT if available, else fallback to first available
    if preferred_ref in network_names:
        network_ref = preferred_ref
    else:
        network_ref = network_names[0]

    ref_idx = np.where(network_names == network_ref)[0][0]

    # Get the reference coordinates
    ref_x = col_coords.iloc[ref_idx, 0]
    ref_y = col_coords.iloc[ref_idx, 1]

    # Flip x-axis if ref_x > 0 (we want it negative for left)
    if ref_x > 0:
        col_coords.iloc[:, 0] *= -1
        row_coords.iloc[:, 0] *= -1

    # Flip y-axis if ref_y < 0 (we want it positive for top)
    if ref_y < 0:
        col_coords.iloc[:, 1] *= -1
        row_coords.iloc[:, 1] *= -1

    fig = go.Figure()

    # Define colors
    base_color = "#E41A1C"        # red (for original networks)
    generated_color = "#377EB8"   # blue (colorblind-friendly distinct color)

    # Compute max distance from origin for column and row coordinates
    col_max = np.max(np.sqrt(col_coords.iloc[:,0]**2 + col_coords.iloc[:,1]**2))
    row_max = np.max(np.sqrt(row_coords.iloc[:,0]**2 + row_coords.iloc[:,1]**2))

    # Scale row_coords so their max distance is slightly less than col_coords
    scale_factor = 0.8 * col_max / row_max  # 0.8 = 80% of column radius
    row_coords_scaled = row_coords.copy()
    row_coords_scaled.iloc[:,0] *= scale_factor
    row_coords_scaled.iloc[:,1] *= scale_factor

    # Scatter plot
    fig = go.Figure()
    # Boolean mask for which points are generated
    is_generated = [("_generated" in name) for name in network_names]

    # Split the data
    x_gen = col_coords.iloc[:, 0][is_generated]
    y_gen = col_coords.iloc[:, 1][is_generated]
    text_gen = [name for name in network_names if "_generated" in name]

    x_base = col_coords.iloc[:, 0][~pd.Series(is_generated)]
    y_base = col_coords.iloc[:, 1][~pd.Series(is_generated)]
    text_base = [name for name in network_names if "_generated" not in name]


    # --- TRACE 1: Generated ---
    fig.add_trace(go.Scatter(
        x=x_gen,
        y=y_gen,
        mode='markers+text',
        text=text_gen,
        hoverinfo='text',
        textposition='top center',
        marker=dict(size=10, color=generated_color),
        name='generated_headlines',
    ))

    # --- TRACE 2: Base (Non-generated) ---
    fig.add_trace(go.Scatter(
        x=x_base,
        y=y_base,
        mode='markers+text',
        text=text_base,
        hoverinfo='text',
        textposition='top center',
        marker=dict(size=10, color=base_color),
        name='headlines',
    ))

    if show_ngrams:
        # Scatter for n-grams
        fig.add_trace(go.Scatter(
            x=row_coords_scaled.iloc[:,0],
            y=row_coords_scaled.iloc[:,1],
            mode='markers',
            text=feature_names,  # We'll annotate top ones manually
            hoverinfo="text",
            marker=dict(size=4, color='blue', opacity=0.2),
            name='n-grams'
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
            showgrid=False,       # removes vertical grid lines
            zeroline=False,       # removes x=0 line
            showticklabels=False, # hides tick labels
            ticks="",              # removes tick marks
        ),
        yaxis=dict(
            showgrid=False,       # removes horizontal grid lines
            zeroline=False,       # removes y=0 line
            showticklabels=False, # hides tick labels
            ticks="",              # removes tick marks
        ),
        margin=dict(l=50, r=50, t=0, b=0),
        width=800,
        height=600
    )
    fig.update_traces(cliponaxis=False)

    return fig

if __name__ == "__main__":
    app.run(debug=True)