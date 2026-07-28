Problem Statement

1. What makes a movie "successful" in this project?

In this project, a movie is considered successful if it generates more revenue than the amount spent on its production. 
Commercial success is measured by comparing the movie's total revenue with its production budget.

Classification Rule:

Successful Movie (Success = 1): Revenue > Budget
Unsuccessful Movie (Success = 0): Revenue ≤ Budget

This rule allows the dataset to be divided into two categories, which can then be analysed to identify the factors that 
contribute to financial success.

2. Why is predicting film success valuable?

Predicting movie success is valuable because it helps stakeholders make informed decisions before investing large amounts 
of money in a film.

Stakeholder 1: Movie Studios

Movie studios can use success predictions to:

Select projects with a higher probability of generating profits.
Allocate production budgets more effectively.
Develop targeted marketing strategies.
Reduce the financial risk associated with unsuccessful movies.
Stakeholder 2: Investors

Investors can benefit by:

Identifying movies with strong profit potential.
Making better investment decisions based on historical data.
Reducing investment risks.
Maximising return on investment (ROI).

Overall, predicting movie success supports better planning, efficient resource allocation, and improved financial outcomes.

3. State the objective of the project and list at least three concrete steps you will take to reach it.
4. 
Project Objective

The objective of this project is to analyse historical movie data to identify the key factors associated with 
commercial success and present the findings through an interactive Streamlit dashboard.

Steps to Achieve the Objective

Step 1: Data Preparation

Load the movie dataset.
Clean the data by handling missing values and removing duplicates.
Create a new target variable (success) based on the relationship between revenue and budget.

Step 2: Exploratory Data Analysis (EDA)

Analyse relationships between budget, revenue, popularity, runtime, and audience ratings.
Explore genre-wise performance.
Create visualisations to identify trends and patterns.

Step 3: Statistical Analysis

Perform Independent Samples T-Tests to compare popularity and vote averages between successful and unsuccessful movies.
Conduct a Chi-Square Test to determine whether movie genre is associated with financial success.

Step 4: Dashboard Development

Build an interactive Streamlit dashboard.
Add filters, KPI cards, charts, statistical results, and business insights.
Enable users to explore movie performance dynamically.

4. This is a classification problem. Explain what that means and what your model's target variable will be.

A classification problem is a type of machine learning or data analysis task where each observation is assigned to one of 
two or more predefined categories. Instead of predicting a continuous numerical value, the goal is to predict the correct 
class or label.

In this project, each movie is classified as either successful or unsuccessful based on its financial performance.

Target Variable

The target variable is success.

It is a binary variable with two possible values:

Success Value	Meaning
1	Movie is successful (Revenue > Budget)
0	Movie is unsuccessful (Revenue ≤ Budget)

This target variable enables the analysis to compare successful and unsuccessful movies and identify the characteristics that are most strongly associated with commercial success.
