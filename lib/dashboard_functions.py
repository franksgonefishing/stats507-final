import plotly.graph_objects as go

# shared function to filter the dataframe shown in the dashboard visuals
def filter_df(df, month_range=None, day_range=None, network_selection=None, news_category_selection=None, headline_contains=None, include_generated_headlines=False):

    if month_range:
        df = df[(df["month"] >= month_range[0]) & (df["month"] <= month_range[1])]
    
    if day_range:
        df = df[(df["day"] >= day_range[0]) & (df["day"] <= day_range[1])]

    if network_selection:
        df = df[df["network"].isin(network_selection)]
    
    if news_category_selection:
        df = df[df["news_category"].isin(news_category_selection)]
    
    if isinstance(headline_contains, str) and headline_contains.strip():
        df = df[df["headline"].str.contains(headline_contains.lower(), case=False)]

    if not include_generated_headlines:
        df = df[df["source"] == "real_headlines"]

    return df


# function to present an error message if something goes wrong with any of the visuals in the dashboard
def error_fig(error_msg):
    fig = go.Figure()

    fig.add_annotation(
        text=error_msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="red")
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False}
    )

    return fig


# return a matrix to provide color split between significant and not significant p-values
def p_val_colors(p_val_cutoff):

    light_blue = "rgb(173,216,230)"   # for p < 0.05
    light_red  = "rgb(255,182,193)"   # for p ≥ 0.05

    return [
        [0.0, light_blue],
        [p_val_cutoff,   light_blue],
        [p_val_cutoff,   light_red],
        [1.0, light_red]
    ]