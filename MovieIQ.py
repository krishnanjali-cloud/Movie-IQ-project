# ================================================================
# MovieIQ - Part 1: Data Preparation & Data Cleaning
# ================================================================

# ------------------------------------------------
# Step 1: Import Required Libraries
# ------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Display all columns
pd.set_option('display.max_columns', None)

# Improve chart appearance
plt.style.use('ggplot')

print("Libraries Imported Successfully!")


# --------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="MovieIQ Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------
# Dashboard Title
# --------------------------------------------------------------

st.title("🎬 MovieIQ Dashboard - Predictive Analytics on Film Success")

st.markdown("""
### Movie Performance Analytics Dashboard

This dashboard analyzes movie performance using
financial, audience and genre information.

The dashboard provides insights into

- Budget
- Revenue
- Popularity
- Runtime
- Vote Average
- Genres
""")

st.markdown("---")

# ------------------------------------------------
# Step 2: Load the Dataset
# ------------------------------------------------
# --- File Upload ---
df1 = st.file_uploader("Upload your movies CSV file", type=["csv"])

if df1:
    df = pd.read_csv(df1)

    print("Dataset Loaded Successfully!")


    # ------------------------------------------------
    # Step 3: Initial Data Exploration
    # ------------------------------------------------
    
    # Display first five rows
    print("\nFirst Five Records")
    print(df.head())
    
    # Display last five rows
    print("\nLast Five Records")
    print(df.tail())
    
    # Shape of dataset
    print("\nDataset Shape")
    print(df.shape)
    
    print(f"\nNumber of Rows : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")
    
    # Display column names
    print("\nColumn Names")
    print(df.columns.tolist())
    
    # Dataset Information
    print("\nDataset Information")
    print(df.info())
    
    # Summary Statistics
    print("\nSummary Statistics")
    print(df.describe())
    
    
    # ------------------------------------------------
    # Step 4: Check Data Types
    # ------------------------------------------------
    
    print("\nData Types")
    print(df.dtypes)
    
    
    # ------------------------------------------------
    # Step 5: Check Missing Values
    # ------------------------------------------------
    
    print("\nMissing Values")
    print(df.isnull().sum())
    
    # Percentage of Missing Values
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    print("\nMissing Value Percentage")
    print(missing_percent.sort_values(ascending=False))
    
    
    # ------------------------------------------------
    # Step 6: Visualize Missing Values (Optional)
    # ------------------------------------------------
    
    plt.figure(figsize=(12,5))
    
    sns.heatmap(df.isnull(),
                cbar=False,
                cmap='viridis')
    
    plt.title("Missing Value Heatmap")
    plt.show()
    
    
    # ------------------------------------------------
    # Step 7: Check Duplicate Records
    # ------------------------------------------------
    
    duplicates = df.duplicated().sum()
    
    print("\nDuplicate Records :", duplicates)
    
    
    # ------------------------------------------------
    # Step 8: Remove Duplicate Records
    # ------------------------------------------------
    
    df.drop_duplicates(inplace=True)
    
    print("\nDataset Shape After Removing Duplicates")
    print(df.shape)
    
    
    # ------------------------------------------------
    # Step 9: Check Budget and Revenue
    # ------------------------------------------------
    
    print("\nMovies with Budget = 0")
    print((df['budget'] == 0).sum())
    
    print("\nMovies with Revenue = 0")
    print((df['revenue'] == 0).sum())
    
    
    # ------------------------------------------------
    # Step 10: Remove Invalid Records
    # ------------------------------------------------
    
    df = df[(df['budget'] > 0) &
            (df['revenue'] > 0)]
    
    print("\nDataset Shape After Removing Invalid Records")
    print(df.shape)
    
    
    # ------------------------------------------------
    # Step 11: Check Remaining Missing Values
    # ------------------------------------------------
    
    print("\nRemaining Missing Values")
    print(df.isnull().sum())
    
    
    # ------------------------------------------------
    # Step 12: Handle Missing Genre Values
    # ------------------------------------------------
    
    df['genres'] = df['genres'].fillna("Unknown")
    
    print("\nMissing Genres Filled Successfully")
    
    
    # ------------------------------------------------
    # Step 13: Feature Engineering
    # Create Success Column
    # ------------------------------------------------
    
    df['success'] = np.where(
        df['revenue'] > df['budget'],
        1,
        0
    )
    
    print("\nSuccess Column Created")
    
    
    # ------------------------------------------------
    # Step 14: Check Class Distribution
    # ------------------------------------------------
    
    print("\nSuccess Distribution")
    print(df['success'].value_counts())
    
    print("\nSuccess Percentage")
    print(round(df['success'].value_counts(normalize=True) * 100, 2))
    
    
    # ------------------------------------------------
    # Step 15: Process Genre Column
    # ------------------------------------------------
    import ast

    # Convert string to Python list
    df["genres"] = df["genres"].apply(ast.literal_eval)

    # Create new column with genre name
    df["genre_name"] = df["genres"].apply(
        lambda x: x[0]["name"] if len(x) > 0 else None
    )
    
    print("\nGenre Column Processed")
    
    
    # ------------------------------------------------
    # Step 16: Final Dataset Information
    # ------------------------------------------------
    
    print("\nFinal Dataset Shape")
    print(df.shape)
    
    print("\nFinal Dataset Information")
    print(df.info())
    
    print("\nFirst Five Records After Cleaning")
    print(df.head())
    
    
    # ------------------------------------------------
    # Step 17: Save Cleaned Dataset
    # ------------------------------------------------
    
    df.to_csv("movies_cleaned.csv", index=False)
    
    print("\nCleaned Dataset Saved Successfully!")
    
    # MovieIQ – Part 2: Exploratory Data Analysis (EDA)
    
    # ==============================================================
    
    import os
    
    plt.style.use("ggplot")
    pd.set_option('display.max_columns', None)
    
    # ------------------------------------------------
    # Step 2 : Load Cleaned Dataset
    # ------------------------------------------------
    
    df = pd.read_csv("movies_cleaned.csv")
    
    # Convert genres back into list format
    df['genres'] = df['genres'].str.replace("[", "", regex=False)
    df['genres'] = df['genres'].str.replace("]", "", regex=False)
    df['genres'] = df['genres'].str.replace("'", "", regex=False)
    df['genres'] = df['genres'].str.split(",")
    
    print(df.head())
    
    
    # Analysis 1 : Dataset Overview
    
    print("Total Movies :", len(df))
    print("Successful Movies :", df['success'].sum())
    print("Failed Movies :", len(df)-df['success'].sum())
    
    print("\nAverage Budget")
    print(df['budget'].mean())
    
    print("\nAverage Revenue")
    print(df['revenue'].mean())
    
    print("\nAverage Runtime")
    print(df['runtime'].mean())
    
    print("\nAverage Vote")
    print(df['vote_average'].mean())
    
    
    # Analysis 2 : Budget vs Revenue
    
    plt.figure(figsize=(10,6))
    
    sns.scatterplot(
        data=df,
        x='budget',
        y='revenue',
        alpha=0.6
    )
    
    plt.title("Budget vs Revenue")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    
    plt.tight_layout()
    os.makedirs("assets",exist_ok=True)
    plt.savefig("assets/budget_vs_revenue.png")
    plt.show()
    
    # Analysis 3 : Genre Distribution
    
    genre_df = df.explode('genres')
    
    genre_count = genre_df['genres'].value_counts()
    
    print(genre_count)
    
    
    plt.figure(figsize=(12,6))
    
    sns.countplot(
        data=genre_df,
        y='genres',
        order=genre_count.index
    )
    
    plt.title("Movie Genre Distribution")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genres")
    
    plt.tight_layout()
    
    plt.savefig("assets/genre_distribution.png")
    
    plt.show()
    
    # Analysis 4 : Success Rate by Genre
    
    genre_success = genre_df.groupby("genres")["success"].mean()
    
    genre_success = genre_success.sort_values(ascending=False)
    
    print(genre_success)
    
    
    plt.figure(figsize=(12,6))
    
    genre_success.plot(
        kind="bar",
        color="steelblue"
    )
    
    plt.title("Success Rate by Genre")
    
    plt.xlabel("Genre")
    
    plt.ylabel("Success Rate")
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.savefig("assets/genre_success.png")
    
    plt.show()
    
    # Analysis 5 : Popularity vs Success
    
    plt.figure(figsize=(8,6))
    
    sns.boxplot(
        x="success",
        y="popularity",
        data=df
    )
    
    plt.title("Popularity vs Movie Success")
    
    plt.xlabel("Success")
    
    plt.ylabel("Popularity")
    
    plt.tight_layout()
    
    plt.savefig("assets/popularity_boxplot.png")
    
    plt.show()
    
    plt.figure(figsize=(8,6))
    
    sns.boxplot(
        x="success",
        y="vote_average",
        data=df
    )
    
    plt.title("Vote Average vs Success")
    
    plt.xlabel("Success")
    
    plt.ylabel("Vote Average")
    
    plt.tight_layout()
    
    plt.savefig("assets/voteaverage_boxplot.png")
    
    plt.show()
    
    # Analysis 7 : Runtime vs Success
    
    plt.figure(figsize=(8,6))
    
    sns.boxplot(
        x="success",
        y="runtime",
        data=df
    )
    
    plt.title("Runtime vs Success")
    
    plt.xlabel("Success")
    
    plt.ylabel("Runtime")
    
    plt.tight_layout()
    
    plt.savefig("assets/runtime_boxplot.png")
    
    plt.show()
    
    # Analysis 8 : Correlation Heatmap
    
    numeric_df = df.select_dtypes(include=['number'])
    
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10,8))
    
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
    )
    
    plt.title("Correlation Heatmap")
    
    plt.tight_layout()
    
    plt.savefig("assets/correlation_heatmap.png")
    
    plt.show()
    
    # Analysis 9 : Revenue Distribution
    
    plt.figure(figsize=(10,6))
    
    sns.histplot(
        df['revenue'],
        bins=40,
        kde=True
    )
    
    plt.title("Revenue Distribution")
    
    plt.xlabel("Revenue")
    
    plt.ylabel("Number of Movies")
    
    plt.tight_layout()
    
    plt.show()
    
    # Analysis 10 : Budget Distribution
    
    plt.figure(figsize=(10,6))
    
    sns.histplot(
        df['budget'],
        bins=40,
        kde=True
    )
    
    plt.title("Budget Distribution")
    
    plt.xlabel("Budget")
    
    plt.ylabel("Number of Movies")
    
    plt.tight_layout()
    
    plt.show()
    
    # Analysis 11 : Average Revenue by Genre
    
    genre_revenue = genre_df.groupby("genres")["revenue"].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12,6))
    
    genre_revenue.plot(kind='bar')
    
    plt.title("Average Revenue by Genre")
    
    plt.xlabel("Genre")
    
    plt.ylabel("Average Revenue")
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.show()
    
    # ================================================================
    # MovieIQ - Part 3 : Statistical Testing
    # ================================================================
    
    
    from scipy.stats import ttest_ind
    from scipy.stats import chi2_contingency
    
    # Convert genres column back to list
    df["genres"] = (
        df["genres"]
        .fillna('')
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.split(",")
    )
    
    print("Dataset Loaded Successfully")
    
    
    # Separate the two groups
    
    successful_movies = df[df["success"] == 1]["popularity"]
    
    unsuccessful_movies = df[df["success"] == 0]["popularity"]
    
    # Perform Independent T-Test
    
    t_statistic, p_value = ttest_ind(
        successful_movies,
        unsuccessful_movies,
        equal_var=False
    )
    
    print("Independent Samples T-Test")
    print("-" * 40)
    
    print("T Statistic :", round(t_statistic,4))
    print("P Value :", round(p_value,6))
    
    # Interpretation
    
    alpha = 0.05
    
    print("\nInterpretation")
    
    if p_value < alpha:
    
        print("Reject Null Hypothesis")
    
        print("Popularity is significantly different between successful and unsuccessful movies.")
    
    else:
    
        print("Fail to Reject Null Hypothesis")
    
        print("Popularity does not significantly differ between the two groups.")
    
    
    # Visualization
    
    plt.figure(figsize=(8,6))
    
    sns.boxplot(
        data=df,
        x="success",
        y="popularity"
    )
    
    plt.title("Popularity Distribution by Movie Success")
    
    plt.xlabel("Movie Success")
    
    plt.ylabel("Popularity")
    
    plt.show()
    
    # Test 2: Independent Samples T-Test (Vote Average vs Success)
    
    successful_vote = df[df["success"] == 1]["vote_average"]
    
    unsuccessful_vote = df[df["success"] == 0]["vote_average"]
    
    t_stat, p_val = ttest_ind(
        successful_vote,
        unsuccessful_vote,
        equal_var=False
    )
    
    print("\nVote Average T-Test")
    print("-" * 40)
    
    print("T Statistic :", round(t_stat,4))
    print("P Value :", round(p_val,6))
    
    # Interpretation
    
    if p_val < 0.05:
    
        print("Audience ratings differ significantly.")
    
    else:
    
        print("Audience ratings are not significantly different.")
    
    
    # Visualization
    
    plt.figure(figsize=(8,6))
    
    sns.boxplot(
        x="success",
        y="vote_average",
        data=df
    )
    
    plt.title("Vote Average by Movie Success")
    
    plt.show()
    
    # Test 3: Chi-Square Test (Genre vs Success)
    
    # Expand genres
    
    genre_df = df.explode("genres","success")
    
    # Remove extra spaces
    
    genre_df["genres"] = genre_df["genres"].str.strip()
    
    # Create contingency table
    #print(len(df['genres']),len(df['success']))
    contingency_table = pd.crosstab(
        genre_df["genres"],
        genre_df["success"]
    )
    
    print(contingency_table.head())
    
    # Perform Chi-Square Test
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\nChi-Square Test")
    print("-" * 40)
    
    print("Chi-Square Statistic :", round(chi2,4))
    print("Degrees of Freedom :", dof)
    print("P Value :", round(p,6))
    
    # Interpretation
    
    if p < 0.05:
    
        print("Reject Null Hypothesis")
    
        print("Movie genre and success are significantly associated.")
    
    else:
    
        print("Fail to Reject Null Hypothesis")
    
        print("No significant association exists.")
    
    
    # Visualization
    
    genre_success = pd.crosstab(
        genre_df["genres"],
        genre_df["success"]
    )
    
    genre_success.plot(
        kind="bar",
        figsize=(12,6)
    )
    
    plt.title("Genre vs Movie Success")
    
    plt.xlabel("Genre")
    
    plt.ylabel("Movie Count")
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.show()
    
    
    # Statistical Summary Table
    
    summary = pd.DataFrame({
    
        "Test":[
            "Popularity T-Test",
            "Vote Average T-Test",
            "Genre Chi-Square"
        ],
    
        "Statistic":[
            t_statistic,
            t_stat,
            chi2
        ],
    
        "P-Value":[
            p_value,
            p_val,
            p
        ]
    
    })
    
    print(summary)
    
    # Export Statistical Results
    
    summary.to_csv(
        "statistical_test_results.csv",
        index=False
    )
    
    print("Statistical Results Saved Successfully")
    
    # Final Business Insights
    
    print("="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    
    print("""
    1. The Independent Samples T-Test evaluates whether popularity differs
       significantly between successful and unsuccessful movies.
    
    2. A second T-Test examines whether audience ratings (vote average)
       are significantly different across the two groups.
    
    3. The Chi-Square Test determines whether movie genre and financial
       success are associated.
    
    4. A p-value below 0.05 indicates that the observed relationship is
       statistically significant and unlikely to have occurred by chance.
    
    5. These statistical findings provide stronger evidence for business
       recommendations than visual observations alone.
    """)
    
    
    # --------------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------------
    
    @st.cache_data
    def load_data():
    
        df = pd.read_csv("movies_cleaned.csv")
    
        df["genres"] = (
            df["genres"]
            .str.replace("[","",regex=False)
            .str.replace("]","",regex=False)
            .str.replace("'","",regex=False)
            .str.split(",")
        )
    
        return df
    
    
    df = load_data()
    
    # --------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------
    
    st.sidebar.header("🎯 Dashboard Filters")
    
    # --------------------------------------------------------------
    # Genre Filter
    # --------------------------------------------------------------
    
    genre_df = df.explode("genre_name") 
    genre_df["genre_name"] = genre_df["genre_name"].str.strip() 
    genre_list = sorted( genre_df["genre_name"].dropna().unique() ) 
    selected_genre = st.sidebar.selectbox( "Select genre", ["All"]+list(genre_list) ) 


    # --------------------------------------------------------------
    # Vote Average Filter
    # --------------------------------------------------------------
        
    min_vote = st.sidebar.slider(
        
        "Minimum Vote Average",

        float(df["vote_average"].min()),

        float(df["vote_average"].max()),

        float(df["vote_average"].min())

    )
        
    # --------------------------------------------------------------
    # Runtime Filter
    # --------------------------------------------------------------

    runtime_range = st.sidebar.slider(
        
        "Runtime (Minutes)",
        
        int(df["runtime"].min()),
        
        int(df["runtime"].max()),
        
        (
            int(df["runtime"].min()),
            int(df["runtime"].max())
        )
        
    )
        
        # --------------------------------------------------------------
        # Popularity Filter
        # --------------------------------------------------------------
        
    min_popularity = st.sidebar.slider(
        
        "Minimum Popularity",
        
        float(df["popularity"].min()),
        
        float(df["popularity"].max()),
        
        float(df["popularity"].min())
        
    )
    
    # --------------------------------------------------------------
    # Apply Filters
    # --------------------------------------------------------------
    
    filtered_df = df.copy()
    
    filtered_df = filtered_df[
        filtered_df["vote_average"] >= min_vote
        ]
        
    filtered_df = filtered_df[
        (filtered_df["runtime"] >= runtime_range[0]) &
        (filtered_df["runtime"] <= runtime_range[1])
        ]
        
    filtered_df = filtered_df[
        filtered_df["popularity"] >= min_popularity
        ]
        
    if selected_genre != "All":
        filtered_df = filtered_df[
                filtered_df["genre_name"].str.contains(
                    selected_genre,
                    case=False,
                    na=False
                )
            ]

    st.dataframe(filtered_df)
    
    # --------------------------------------------------------------
    # Dashboard Metrics
    # --------------------------------------------------------------
    
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # --------------------------------------------------------------
    # KPI 1
    # --------------------------------------------------------------
    
    with col1:
    
        st.metric(
    
            label="🎥 Total Movies",
    
            value=len(filtered_df)
    
        )
    
    # --------------------------------------------------------------
    # KPI 2
    # --------------------------------------------------------------
    
    with col2:
    
        successful = filtered_df["success"].sum()
    
        st.metric(
    
            label="✅ Successful Movies",
    
            value=int(successful)
    
        )
    
    # --------------------------------------------------------------
    # KPI 3
    # --------------------------------------------------------------
    
    with col3:
    
        failed = len(filtered_df) - successful
    
        st.metric(
    
            label="❌ Failed Movies",
    
            value=int(failed)
    
        )
    
    # --------------------------------------------------------------
    # KPI 4
    # --------------------------------------------------------------
    
    with col4:
    
        success_rate = (
            successful /
            len(filtered_df)
        ) * 100 if len(filtered_df) > 0 else 0
    
        st.metric(
    
            label="📈 Success Rate",
    
            value=f"{success_rate:.2f}%"
    
        )
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # Financial KPIs
    # --------------------------------------------------------------
    
    st.header("💰 Financial Summary")
    
    financial1, financial2, financial3 = st.columns(3)
    
    with financial1:
    
        st.metric(
    
            "Average Budget",
    
            f"${filtered_df['budget'].mean():,.0f}"
    
        )
    
    with financial2:
    
        st.metric(
    
            "Average Revenue",
    
            f"${filtered_df['revenue'].mean():,.0f}"
    
        )
    
    with financial3:
    
        profit = (
            filtered_df["revenue"] -
            filtered_df["budget"]
        ).mean()
    
        st.metric(
    
            "Average Profit",
    
            f"${profit:,.2f}"
    
        )
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # Audience KPIs
    # --------------------------------------------------------------
    
    st.header("⭐ Audience Metrics")
    
    audience1, audience2, audience3 = st.columns(3)
    
    with audience1:
    
        st.metric(
    
            "Average Vote",
    
            round(
                filtered_df["vote_average"].mean(),
                2
            )
    
        )
    
    with audience2:
    
        st.metric(
    
            "Average Popularity",
    
            round(
                filtered_df["popularity"].mean(),
                2
            )
    
        )
    
    with audience3:
    
        st.metric(
    
            "Average Runtime",
    
            round(
                filtered_df["runtime"].mean(),
                1
            )
    
        )
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # Display Filtered Dataset
    # --------------------------------------------------------------
    
    st.header("📄 Filtered Dataset")
    
    st.dataframe(
    
        filtered_df,
    
        use_container_width=True
    
    )
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # Download Filtered Dataset
    # --------------------------------------------------------------
    
    csv = filtered_df.to_csv(index=False)
    
    st.download_button(
    
        label="⬇ Download Filtered Dataset",
    
        data=csv,
    
        file_name="Filtered_Movies.csv",
    
        mime="text/csv"
    
    )
    
    st.success("Dashboard Loaded Successfully!")
    
    # =====================================================
    # Budget vs Revenue Scatter Plot
    # =====================================================
    
    st.header("📊 Budget vs Revenue Analysis")
    
    fig1, ax1 = plt.subplots(figsize=(10,6))
    
    sns.scatterplot(
        data=filtered_df,
        x="budget",
        y="revenue",
        hue="success",
        palette="Set1",
        alpha=0.7,
        ax=ax1
    )
    
    ax1.set_title("Budget vs Revenue")
    
    ax1.set_xlabel("Budget")
    
    ax1.set_ylabel("Revenue")
    
    st.pyplot(fig1)
    
    st.markdown("""
    
    ### Business Insight
    
    - Each point represents one movie.
    - Movies above the break-even line generally generated profit.
    - Higher-budget movies often generate higher revenues.
    - Some low-budget movies become blockbuster successes.
    - Some high-budget movies fail to recover their investment.
    
    """)
    
    st.markdown("---")
    
    # Section 2: Revenue Distribution
    
    # =====================================================
    # Revenue Distribution
    # =====================================================
    
    st.header("💰 Revenue Distribution")
    
    fig2, ax2 = plt.subplots(figsize=(10,5))
    
    sns.histplot(
        filtered_df["revenue"],
        bins=35,
        kde=True,
        color="green",
        ax=ax2
    )
    
    ax2.set_title("Distribution of Revenue")
    
    ax2.set_xlabel("Revenue")
    
    ax2.set_ylabel("Number of Movies")
    
    st.pyplot(fig2)
    
    st.markdown("""
    
    ### Business Insight
    
    - Shows how movie revenues are distributed.
    - Most movies generate relatively low revenues.
    - Only a small number become extremely high-grossing blockbusters.
    - The distribution is typically right-skewed because of a few exceptionally successful films.
    
    """)
    
    st.markdown("---")
    
    # Section 3: Budget Distribution
    
    # =====================================================
    # Budget Distribution
    # =====================================================
    
    st.header("💵 Budget Distribution")
    
    fig3, ax3 = plt.subplots(figsize=(10,5))
    
    sns.histplot(
        filtered_df["budget"],
        bins=35,
        kde=True,
        color="royalblue",
        ax=ax3
    )
    
    ax3.set_title("Distribution of Movie Budgets")
    
    ax3.set_xlabel("Budget")
    
    ax3.set_ylabel("Number of Movies")
    
    st.pyplot(fig3)
    
    st.markdown("""
    
    ### Business Insight
    
    - Displays the investment distribution across movies.
    - Most films are produced with moderate budgets.
    - Only a small number of movies receive exceptionally high production budgets.
    - This helps identify common investment levels within the industry.
    
    """)
    
    st.markdown("---")
    
    # Section 4: Financial Summary Table
    # =====================================================
    # Financial Summary
    # =====================================================
    
    st.header("📋 Financial Summary")
    
    financial_summary = pd.DataFrame({
    
        "Metric":[
    
            "Average Budget",
            "Average Revenue",
            "Maximum Budget",
            "Maximum Revenue",
            "Minimum Budget",
            "Minimum Revenue"
    
        ],
    
        "Value":[
    
            filtered_df["budget"].mean(),
            filtered_df["revenue"].mean(),
            filtered_df["budget"].max(),
            filtered_df["revenue"].max(),
            filtered_df["budget"].min(),
            filtered_df["revenue"].min()
    
        ]
    
    })
    
    st.dataframe(
        financial_summary,
        use_container_width=True
    )
    
    # =====================================================
    # Genre Distribution
    # =====================================================
    
    st.header("🎭 Genre Distribution")
    
    # Create a separate dataframe for genre analysis
    genre_df = filtered_df.explode("genre_name")
    
    # Remove leading/trailing spaces
    genre_df["genre_name"] = genre_df["genre_name"].str.strip()
    
    # Count movies by genre
    genre_count = (
        genre_df["genre_name"]
        .value_counts()
        .sort_values(ascending=False)
    )
    
    fig4, ax4 = plt.subplots(figsize=(10,6))
    
    sns.barplot(
        x=genre_count.values,
        y=genre_count.index,
        palette="viridis",
        ax=ax4
    )
    
    ax4.set_title("Number of Movies by Genre")
    ax4.set_xlabel("Number of Movies")
    ax4.set_ylabel("Genre")
    
    st.pyplot(fig4)
    
    st.markdown("""
    ### Business Insight
    
    - Identifies the most frequently produced movie genres.
    - Helps understand industry production trends.
    - Genres with the highest counts indicate greater production focus.
    """)
    
    st.markdown("---")
    
    # =====================================================
    # Success Rate by Genre
    # =====================================================
    
    st.header("🏆 Success Rate by Genre")
    
    genre_success = (
        genre_df
        .groupby("genre_name")["success"]
        .mean()
        .sort_values(ascending=False)
    )
    
    fig5, ax5 = plt.subplots(figsize=(10,6))
    
    sns.barplot(
        x=genre_success.values,
        y=genre_success.index,
        palette="Set2",
        ax=ax5
    )
    
    ax5.set_title("Success Rate by Genre")
    
    ax5.set_xlabel("Success Rate")
    
    ax5.set_ylabel("Genre")
    
    st.pyplot(fig5)
    
    st.markdown("""
    ### Business Insight
    
    - Shows the percentage of successful movies within each genre.
    - Helps identify genres with consistently strong financial performance.
    - Useful for production companies when selecting future projects.
    """)
    
    st.markdown("---")
    
    # =====================================================
    # Average Revenue by Genre
    # =====================================================
    
    st.header("💲 Average Revenue by Genre")
    
    genre_revenue = (
    
        genre_df
    
        .groupby("genre_name")["revenue"]
    
        .mean()
    
        .sort_values(ascending=False)
    
    )
    
    fig6, ax6 = plt.subplots(figsize=(10,6))
    
    sns.barplot(
    
        x=genre_revenue.values,
    
        y=genre_revenue.index,
    
        palette="rocket",
    
        ax=ax6
    
    )
    
    ax6.set_title("Average Revenue by Genre")
    
    ax6.set_xlabel("Average Revenue")
    
    ax6.set_ylabel("Genre")
    
    st.pyplot(fig6)
    
    st.markdown("""
    
    ### Business Insight
    
    - Compares average revenue generated by each genre.
    - Reveals which genres generate the greatest financial returns.
    - Helps investors and studios prioritize profitable genres.
    
    """)
    
    st.markdown("---")
    
    # =====================================================
    # Genre Summary Table
    # =====================================================
    
    st.header("📋 Genre Performance Summary")
    
    genre_summary = (
    
        genre_df
    
        .groupby("genre_name")
    
        .agg(
    
            Total_Movies=("title","count"),
    
            Successful_Movies=("success","sum"),
    
            Average_Revenue=("revenue","mean"),
    
            Average_Budget=("budget","mean"),
    
            Average_Rating=("vote_average","mean"),
    
            Average_Popularity=("popularity","mean")
    
        )
    
    )
    
    genre_summary["Success_Rate (%)"] = (
    
        genre_summary["Successful_Movies"]
    
        /
    
        genre_summary["Total_Movies"]
    
    ) * 100
    
    genre_summary = genre_summary.sort_values(
    
        by="Success_Rate (%)",
    
        ascending=False
    
    )
    
    st.dataframe(
    
        genre_summary,
    
        use_container_width=True
    
    )
    
    # =====================================================
    # Top Revenue Genres
    # =====================================================
    
    st.header("🥇 Top Revenue Generating Genres")
    
    top_revenue = (
    
        genre_summary
    
        .sort_values(
    
            by="Average_Revenue",
    
            ascending=False
    
        )
    
        .head(10)
    
    )
    
    st.table(top_revenue[
    
        [
    
            "Average_Revenue",
    
            "Success_Rate (%)"
    
        ]
    
    ])
    
    # =====================================================
    # Most Successful Genres
    # =====================================================
    
    st.header("⭐ Highest Success Rate Genres")
    
    top_success = (
    
        genre_summary
    
        .sort_values(
    
            by="Success_Rate (%)",
    
            ascending=False
    
        )
    
        .head(10)
    
    )
    
    st.table(
    
        top_success[
    
            [
    
                "Success_Rate (%)",
    
                "Average_Revenue"
    
            ]
    
        ]
    
    )
    
    
    
    # =====================================================
    # Popularity vs Movie Success
    # =====================================================
    
    st.header("🔥 Popularity Analysis")
    
    fig7, ax7 = plt.subplots(figsize=(10,6))
    
    sns.boxplot(
        data=filtered_df,
        x="success",
        y="popularity",
        palette="Set2",
        ax=ax7
    )
    
    ax7.set_title("Popularity vs Movie Success")
    ax7.set_xlabel("Movie Success")
    ax7.set_ylabel("Popularity")
    
    ax7.set_xticklabels([
        "Not Successful",
        "Successful"
    ])
    
    st.pyplot(fig7)
    
    st.markdown("""
    ### Business Insight
    
    • Successful movies generally have higher popularity scores.
    
    • A higher popularity score usually indicates stronger audience interest.
    
    • Popularity is one of the strongest indicators of commercial success.
    """)
    
    st.markdown("---")
    
    # Section 2: Vote Average vs Movie Success
    
    # =====================================================
    # Vote Average Analysis
    # =====================================================
    
    st.header("⭐ Audience Rating Analysis")
    
    fig8, ax8 = plt.subplots(figsize=(10,6))
    
    sns.boxplot(
        data=filtered_df,
        x="success",
        y="vote_average",
        palette="coolwarm",
        ax=ax8
    )
    
    ax8.set_title("Vote Average vs Movie Success")
    ax8.set_xlabel("Movie Success")
    ax8.set_ylabel("Vote Average")
    
    ax8.set_xticklabels([
        "Not Successful",
        "Successful"
    ])
    
    st.pyplot(fig8)
    
    st.markdown("""
    
    ### Business Insight
    
    • Highly rated movies generally perform better financially.
    
    • Audience satisfaction contributes to long-term revenue generation.
    
    • Better ratings often increase word-of-mouth marketing.
    
    """)
    
    st.markdown("---")
    
    # Section 3: Runtime vs Movie Success
    
    # =====================================================
    # Runtime Analysis
    # =====================================================
    
    st.header("⏱ Runtime Analysis")
    
    fig9, ax9 = plt.subplots(figsize=(10,6))
    
    sns.boxplot(
        data=filtered_df,
        x="success",
        y="runtime",
        palette="viridis",
        ax=ax9
    )
    
    ax9.set_title("Runtime vs Movie Success")
    
    ax9.set_xlabel("Movie Success")
    
    ax9.set_ylabel("Runtime (Minutes)")
    
    ax9.set_xticklabels([
        "Not Successful",
        "Successful"
    ])
    
    st.pyplot(fig9)
    
    st.markdown("""
    
    ### Business Insight
    
    • Compare movie durations between successful and unsuccessful films.
    
    • Identify whether longer movies tend to perform better financially.
    
    • Helps understand audience preferences for movie duration.
    
    """)
    
    st.markdown("---")
    
    # Section 4: Correlation Heatmap
    # =====================================================
    # Correlation Heatmap
    # =====================================================
    
    st.header("📈 Correlation Heatmap")
    
    numeric_df = filtered_df.select_dtypes(include=["number"])
    
    correlation = numeric_df.corr()
    
    fig10, ax10 = plt.subplots(figsize=(10,8))
    
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        fmt=".2f",
        ax=ax10
    )
    
    ax10.set_title("Correlation Between Numerical Variables")
    
    st.pyplot(fig10)
    
    st.markdown("""
    
    ### Business Insight
    
    The heatmap helps identify:
    
    - Strong positive relationships
    - Strong negative relationships
    - Weak relationships
    - Potential multicollinearity
    
    A high positive correlation between Budget and Revenue generally indicates that higher-budget movies tend to generate higher revenues.
    
    """)
    
    st.markdown("---")
    
    # Section 5: Correlation Table
    
    # =====================================================
    # Correlation Matrix
    # =====================================================
    
    st.header("📋 Correlation Matrix")
    
    st.dataframe(
        correlation.round(2),
        use_container_width=True
    )
    
    
    # Section 6: Audience Performance Summary
    
    # =====================================================
    # Audience Metrics Summary
    # =====================================================
    
    st.header("🎯 Audience Performance Summary")
    
    summary = pd.DataFrame({
    
        "Metric":[
            "Average Popularity",
            "Maximum Popularity",
            "Average Vote",
            "Maximum Vote",
            "Average Runtime",
            "Maximum Runtime"
        ],
    
        "Value":[
    
            filtered_df["popularity"].mean(),
    
            filtered_df["popularity"].max(),
    
            filtered_df["vote_average"].mean(),
    
            filtered_df["vote_average"].max(),
    
            filtered_df["runtime"].mean(),
    
            filtered_df["runtime"].max()
    
        ]
    
    })
    
    st.dataframe(
        summary,
        use_container_width=True
    )
    
    
    # Section 7: Top Rated Movies
    
    # =====================================================
    # Top Rated Movies
    # =====================================================
    
    st.header("🏆 Top 10 Highest Rated Movies")
    
    top_rated = (
    
        filtered_df
    
        .sort_values(
            by="vote_average",
            ascending=False
        )
    
        [[
            "title",
            "vote_average",
            "revenue",
            "budget",
            "popularity"
        ]]
    
        .head(10)
    
    )
    
    st.dataframe(
        top_rated,
        use_container_width=True
    )
    
    
    # Section 8: Most Popular Movies
    # =====================================================
    # Most Popular Movies
    # =====================================================
    
    st.header("🔥 Top 10 Most Popular Movies")
    
    popular_movies = (
    
        filtered_df
    
        .sort_values(
            by="popularity",
            ascending=False
        )
    
        [[
            "title",
            "popularity",
            "vote_average",
            "revenue",
            "budget"
        ]]
    
        .head(10)
    
    )
    
    st.dataframe(
        popular_movies,
        use_container_width=True
    )
    
    
    """ After completing this section, your dashboard will include:
    
    * **Popularity vs Movie Success** box plot.
    * **Vote Average vs Movie Success** box plot.
    * **Runtime vs Movie Success** box plot.
    * **Interactive Correlation Heatmap** with correlation matrix.
    * **Audience Performance Summary** table.
    * **Top 10 Highest Rated Movies** table.
    * **Top 10 Most Popular Movies** table.
    
    All charts and tables respond dynamically to the sidebar filters, giving users an interactive way to explore audience behavior, movie quality metrics, and relationships between numerical features. Together with Parts **4.1**, **4.2.1**, and **4.2.2**, this forms a complete professional Streamlit dashboard for your MovieIQ Data Analytics project."""
    
    
    # MovieIQ – Part 4.3: Dashboard Footer, Statistical Results, Business Insights & Export
    
    
    """ This final section completes the dashboard by adding:
    
    * Statistical Test Results
    * Business Insights
    * Download Statistical Results
    * Dashboard Footer
    * Dataset Information
    * Conclusion
    * Export Button """
    
    # Section 1: Statistical Test Results
    # ==========================================================
    # Statistical Test Results
    # ==========================================================
    
    st.markdown("---")
    
    st.header("📊 Statistical Test Results")
    
    try:
    
        stat_df = pd.read_csv("statistical_test_results.csv")
    
        st.dataframe(
            stat_df,
            use_container_width=True
        )
    
    except:
    
        st.warning(
            "Statistical Results file not found."
        )
    
    # Section 2: Interpretation of Statistical Tests
    
    st.subheader("📖 Interpretation")
    
    st.info("""
    
    ### Independent Samples T-Test
    
    Purpose
    
    Determine whether popularity and vote average
    are significantly different between successful
    and unsuccessful movies.
    
    Decision Rule
    
    • p-value < 0.05
    
    Reject Null Hypothesis
    
    • p-value > 0.05
    
    Fail to Reject Null Hypothesis
    
    
    -------------------------------------------
    
    ### Chi-Square Test
    
    Purpose
    
    Determine whether movie genre influences
    movie success.
    
    Decision Rule
    
    • p-value < 0.05
    
    Genre and Success are associated.
    
    • p-value > 0.05
    
    No significant relationship exists.
    
    """)
    
    # Section 3: Business Insights
    
    st.markdown("---")
    
    st.header("💼 Business Insights")
    
    st.success("""
    
    1️⃣ Higher budget movies generally generate
    higher revenues.
    
    2️⃣ Popularity has a strong relationship
    with movie success.
    
    3️⃣ Higher audience ratings usually lead
    to better financial performance.
    
    4️⃣ Certain genres consistently outperform
    others.
    
    5️⃣ Budget alone does not guarantee success.
    
    6️⃣ Audience engagement is one of the
    strongest indicators of commercial success.
    
    7️⃣ Production companies should consider
    both financial investment and audience
    interest before launching a movie.
    
    """)
    
    # Section 4: Recommendations
    
    st.markdown("---")
    
    st.header("📌 Business Recommendations")
    
    recommendations = pd.DataFrame({
    
    "Recommendation":[
    
    "Invest in high-performing genres",
    
    "Improve movie marketing",
    
    "Monitor audience popularity",
    
    "Maintain production quality",
    
    "Optimize production budget",
    
    "Focus on audience ratings"
    
    ],
    
    "Business Value":[
    
    "Higher Return on Investment",
    
    "Increase Visibility",
    
    "Increase Revenue",
    
    "Better Customer Satisfaction",
    
    "Reduce Financial Risk",
    
    "Improve Movie Success"
    
    ]
    
    })
    
    st.dataframe(
    
    recommendations,
    
    use_container_width=True
    
    )
    
    # Section 5: Dataset Summary
    
    st.markdown("---")
    
    st.header("📋 Dataset Summary")
    
    summary = pd.DataFrame({
    
    "Metric":[
    
    "Total Movies",
    
    "Successful Movies",
    
    "Failed Movies",
    
    "Average Budget",
    
    "Average Revenue",
    
    "Average Vote",
    
    "Average Popularity",
    
    "Average Runtime"
    
    ],
    
    "Value":[
    
    len(filtered_df),
    
    filtered_df["success"].sum(),
    
    len(filtered_df)-filtered_df["success"].sum(),
    
    round(filtered_df["budget"].mean(),2),
    
    round(filtered_df["revenue"].mean(),2),
    
    round(filtered_df["vote_average"].mean(),2),
    
    round(filtered_df["popularity"].mean(),2),
    
    round(filtered_df["runtime"].mean(),2)
    
    ]
    
    })
    
    st.table(summary)
    
    # Section 6: Download Statistical Results
    
    st.markdown("---")
    
    st.header("⬇ Download Statistical Results")
    
    try:
    
        csv = stat_df.to_csv(index=False)
    
        st.download_button(
    
            label="Download Statistical Results",
    
            data=csv,
    
            file_name="Statistical_Test_Results.csv",
    
            mime="text/csv"
    
        )
    
    except:
    
        pass
    
    # Section 7: Download Entire Filtered Dataset
    
    st.header("⬇ Download Dashboard Dataset")
    
    csv = filtered_df.to_csv(index=False)
    
    st.download_button(
    
    label="Download Dataset",
    
    data=csv,
    
    file_name="MovieIQ_Filtered_Data.csv",
    
    mime="text/csv"
    
    )
    
    # Section 8: Dashboard Conclusion
    
    st.markdown("---")
    
    st.header("🎯 Project Conclusion")
    
    st.write("""
    
    MovieIQ demonstrates how historical movie
    data can be analyzed to understand the key
    drivers of commercial success.
    
    The dashboard combines
    
    ✔ Financial Analysis
    
    ✔ Audience Analysis
    
    ✔ Genre Analysis
    
    ✔ Statistical Testing
    
    into a single interactive application.
    
    The project enables studios, investors and
    business analysts to make informed decisions
    based on historical trends instead of
    assumptions.
    
    """)
    
    # Section 9: About the Dashboard
    
    st.markdown("---")
    
    st.header("ℹ About MovieIQ")
    
    st.write("""
    
    MovieIQ is an end-to-end Data Analytics project.
    
    Technologies Used
    
    • Python
    
    • Pandas
    
    • NumPy
    
    • Matplotlib
    
    • Seaborn
    
    • SciPy
    
    • Streamlit
    
    Project Modules
    
    ✔ Data Cleaning
    
    ✔ Feature Engineering
    
    ✔ Exploratory Data Analysis
    
    ✔ Statistical Testing
    
    ✔ Dashboard Development
    
    """)
    
    # Section 10: Dashboard Footer
    
    st.markdown("---")
    
    st.markdown(
    """
    <center>
    
    ### 🎬 MovieIQ Dashboard
    
    Developed using Python & Streamlit
    
    © 2026 MovieIQ Analytics
    
    </center>
    """,
    unsafe_allow_html=True
    )
    
    
    """ After completing this final section, your dashboard will include:
    
    ### Dashboard Features
    
    * Professional title and landing page.
    * Sidebar filters.
    * KPI cards.
    * Financial analysis charts.
    * Genre analysis charts.
    * Audience analysis charts.
    * Correlation heatmap.
    * Statistical test results.
    * Business insights.
    * Business recommendations.
    * Dataset summary.
    * Download buttons for:
    
      * Filtered dataset
      * Statistical test results
    * Project conclusion.
    * About section.
    * Professional dashboard footer. """
    
    
    
    """ * Data Cleaning & Preparation
    * Exploratory Data Analysis (EDA)
    * Statistical Testing (T-Test & Chi-Square)
    * Interactive filtering
    * Financial, Genre, and Audience analytics
    * Business insights and recommendations
    * Downloadable reports
    
    This results in a portfolio-ready, industry-style Data Analytics project that showcases Python, Pandas, Seaborn, Matplotlib, SciPy, and Streamlit skills suitable for Data Analyst roles.
     """
