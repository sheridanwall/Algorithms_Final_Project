# Algorithms_Final_Project
Final project investigating NOPD misconduct complaints Investigating bias in NOPD misconduct complaints: Methodology & Results Sheridan Wall & Bianca Pallaro

04/07/2021

The purpose of this project was to investigate misconduct complaints filed against officers at the New Orleans Police Department (NOPD). The dataset was published on the City of New Orleans’ Open Data Portal, and it includes both citizen and employee-originated reports from 2016 to 2021. Among other factors, the data is disaggregated by race, age, gender, agency and description of violation.We wanted to determine whether the department’s disciplinary council, the Public Integrity Bureau (PIB), is biased against officers of racial and ethnic minorities in its investigation process. In other words, is the NOPD’s PIB racist?

To answer our question, we used a multivariable logistic regression to predict if an officer’s race, accounting for gender, age and the origin of the complaint, influenced the PIB’s judgment. We started by cleaning each feature, which involved converting inconsistent data into standard categories or null values and changing the data type as necessary. We also created a new column with officer ages divided into bins to compare the effects of different age groups on the investigation’s outcome.

The “Disposition” column in the dataset refers to the outcome of the complaint investigation, and we reviewed the NOPD’s Operations Manual to understand the meaning of each possible result. Allegations receive in a sustained disposition when “the investigation determines by a preponderance of the evidence that the alleged misconduct did occur.” We then eliminated the rows that were classified as pending and created a new column categorizing each disposition as either “Sustained” or “Other.”

Next, we used the statsmodels module to build our logistic regression, which predicted the “Sustained” feature based on the officer’s race and ethnicity, controlling for the officer’s age, gender and the complaint’s origin. To generate a simpler model, we created a new column called “Minority,” which categorized each officer as either a white person (“W”) or a person from a minority racial or ethnic community (“M”). For each feature, we set a reference category: We compared minorities to white, gender to male, age to 25-38 and complaint origin to “Public Initiated.”

While we originally set out to determine racial bias, the most interesting finding was related to the origin of the complaint. The regression shows that, controlling for the officer’s age, minority status and gender, a complaint filed by an employee is about four times more likely to result in a sustained disposition than a complaint filed by a member of the public.

The regression also indicates that, controlling for other features, men are 20% more likely to receive a sustained disposition than women; officers of racial and ethnic minorities are a little over 1% more likely to receive a sustained disposition than white officers; and officers in the 55-69 age group are about 5% more likely to receive a sustained disposition when compared to officers in the 25-38 age group. The odds ratios of other age bins were less significant.

For both the gender and complaint origin features, the p-values are less than 0.05, which is the standard threshold for indicating statistical significance. However, the p-values for age and minority are much higher, suggesting that neither of these features are statistically relevant. Therefore, the regression does not prove a bias against officers of racial or ethnic minorities.

Since missing values may sometimes affect the result of a regression, we eliminated all of the rows with missing values and created a new dataframe. We then performed the above regression on the new dataframe, but both the odds ratios and p-values remained mostly unchanged.

Data Sources: https://catalog.data.gov/dataset/nopd-misconduct-complaints, https://www.nola.gov/getattachment/NOPD/Policies/Chapter-52-1-1-Misconduct-Intake-and-Complaint-Investigation-EFFECTIVE-3-18-18.pdf/
