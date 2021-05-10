#!/usr/bin/env python
# coding: utf-8

# # Final Project - NOPD Misconduct Complaints
# 
# Source: City of New Orleans Open Data, https://catalog.data.gov/dataset/nopd-misconduct-complaints
# 
# 

# In[1]:


# Reading & cleaning the data


# In[2]:


import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 100)


# In[3]:


df = pd.read_csv('NOPD_Misconduct_Complaints.csv')
df.head()


# In[4]:


df['Date Complaint Investigation Complete'] = pd.to_datetime(df["Date Complaint Investigation Complete"], format='%Y-%m-%d')


# In[5]:


df['year_complete'] = df['Date Complaint Investigation Complete'].dt.year


# In[6]:


df['complainant_race_clean'] = df["Complainant Ethnicity"].replace({
    'B':'Black',
    'b' : 'Black',
    'w' : 'White',
    'W':'White',
    'BLACK':'Black',
    'white':'White',
    'black':'Black',
    'Race-Unknown': np.nan,
    'Unknown': np.nan,
    'Unkown':np.nan
})
df.complainant_race_clean.value_counts()


# In[7]:


df['officer_race_clean'] = df['Officer Race Ethnicity'].replace({
    'Black':'Black',
    'White':'White',
    'Hispanic':'Hispanic',
    'Asian/Pacifi':'Asian',
    'Not Specifie':np.nan,
    'Race-Unknown':np.nan,
    'American Ind':'Indigenous',
    'Asian/Pacif':'Asian',
    ' Giving Anything of Value':np.nan,
    'PARAGRAPH 01 - Professionalism':np.nan  
})
df.officer_race_clean.value_counts()


# In[8]:


df['officer_age_clean'] = df['Officer Age'].replace({
    '-38': np.nan,
    '-8': np.nan,
    'Female': np.nan,
    'Male': np.nan
})
# df['officer_age_clean'].value_counts()


# In[9]:


df['officer_age_clean'] = df['officer_age_clean'].astype(float)


# In[10]:


df['officer_gender_clean'] = df['Officer Gender'].replace({
    'Male':'Male',
    'Female':'Female',
    'N': np.nan,
    'Black': np.nan,
    'White': np.nan
})
df.officer_gender_clean.value_counts()


# In[11]:


df['incident_type'] = df['Incident Type']
df.incident_type.value_counts()


# In[12]:


df['minority'] = df['officer_race_clean'].replace({
    'Black':'M',
    'White':'W',
    'Hispanic':'M',
    'Asian':'M',
    'Indigenous':'M'
})
df.minority.value_counts()


# In[13]:


#df.head()


# # Brainstorming the regression
# 
# Definition of each disposition (pg. 18): https://www.nola.gov/getattachment/NOPD/Policies/Chapter-52-1-1-Misconduct-Intake-and-Complaint-Investigation-EFFECTIVE-3-18-18.pdf/
# 
# - Is the police department's discplinary board racist? 
#      - Were complaints filed against Black officers marked as sustained more often than others? 
#          - Controlling for gender, age? 7th district? Year?
#          - Public initiated vs. rank initiated? Does who made the complaint factor into the decision? Is the Public more likely to accuse a Black officer?
#          - Control for race of the complainant
#      - Are they dismissing complaints against white police officers more often than Black?
#      - Use unfounded vs. sustained?
#        - Unfounded—the investigation determines by a preponderance of the evidence that the alleged misconduct did not occur or did not involve the accused officer.
#        - Sustained—the investigation determines by a preponderance of the evidence that the alleged misconduct did occur.
# - Are Black officers more often reported for "serious" offenses? By who (public vs. rank)? Are complaints against Black officers more likely to result in a sustained conviction?

# In[14]:


# df.Disposition.value_counts()


# In[15]:


# df['Division of Complainant'].value_counts()


# In[16]:


# df.year_complete.value_counts()


# In[17]:


# df['incident_type'].value_counts()


# In[18]:


# df['Officer Race Ethnicity'].value_counts()


# In[19]:


# df['Complainant Ethnicity'].value_counts()


# In[20]:


# df['Officer Age'].value_counts().tail(40)


# # Preparing DataFrame for logistic regression

# In[21]:


df2 = df[df['Disposition'] != 'Pending']
# df2.shape


# In[22]:


df2.Disposition.value_counts()


# In[23]:


df2['dispostion_new'] = df2.Disposition.replace({
    'Sustained':'S',
    'Unfounded': 'O',
    'Not Sustained':'O',
    'Other':'O',
    'Exonerated':'O',
    'NFIM':'O',
    'Withdrawn - Mediation':'O',
    'Negotiated Settlement':'O',
    'Resigned under investigation':'O'
})


# In[24]:


# df2.head()


# In[25]:


df2.dispostion_new.value_counts()


# In[26]:


df2['sustained'] = df2.dispostion_new.replace({'S':1,'O': 0})
df2.head()


# In[27]:


labels = [
    'under 25',
    '25-38',
    '39-54',
    '55-69',
    'over 70'
]
breaks = [0, 25, 39, 55, 70, 999]
df2['officer_age_bin'] = pd.cut(df2['officer_age_clean'], bins=breaks, labels=labels)
df2.head()


# In[28]:


df2.officer_age_bin.value_counts()


# In[29]:


new_df = df2.drop(columns = ['Incident Type', 'Date Complaint Received by NOPD (PIB)', 'Complaint classification',
                  'Bureau of Complainant','Division of Complainant','Unit of Complainant','Date Complaint Occurred',
                  'Unit Additional Details of Complainant','Working Status of Complainant','Shift of Complainant',
                 'Unique Officer Allegation ID','Officer Race Ethnicity','Officer Age','Officer years of service',
                 'Officer Gender','Complainant Gender','Complainant Ethnicity','Complainant Age'])


# In[30]:


new_df = new_df.dropna()
new_df.shape


# In[31]:


new_df.head()


# In[ ]:





# In[ ]:





# # Testing logistic regressions

# In[33]:


model = smf.logit("""
    sustained ~ 
        C(officer_race_clean, Treatment('White'))       
""", data=df2)
results = model.fit()
results.summary()


# In[34]:


coefs = pd.DataFrame({
    'coef': results.params.values,
    'odds ratio': np.exp(results.params.values),
    'pvalue': results.pvalues,
    'name': results.params.index
})
coefs


# In[35]:


df2.head()


# In[36]:


model = smf.logit("""
    sustained ~ 
        C(minority, Treatment('W'))
        + C(officer_gender_clean, Treatment('Female'))
        + C(incident_type, Treatment('Public Initiated'))
""", data=df2)
results = model.fit()
results.summary()


# In[37]:


coefs = pd.DataFrame({
    'coef': results.params.values,
    'odds ratio': np.exp(results.params.values),
    'pvalue': results.pvalues,
    'name': results.params.index
})
coefs


# In[42]:


model = smf.logit("""
    sustained ~ 
        C(minority, Treatment('W'))
        + C(officer_gender_clean, Treatment('Male'))
        + C(incident_type, Treatment('Public Initiated'))
        + C(officer_age_bin, Treatment('25-38'))
""", data=df2)
results = model.fit()
results.summary()


# In[43]:


coefs = pd.DataFrame({
    'coef': results.params.values,
    'odds ratio': np.exp(results.params.values),
    'pvalue': results.pvalues,
    'name': results.params.index
})
coefs


# # Testing the regression with new dataframe

# In[ ]:


model = smf.logit("""
    sustained ~ 
        C(minority, Treatment('W'))
        + C(officer_gender_clean, Treatment('Male'))
        + C(incident_type, Treatment('Public Initiated'))
        + C(officer_age_bin, Treatment('25-38'))
""", data=new_df)
results = model.fit()
results.summary()


# In[ ]:


coefs = pd.DataFrame({
    'coef': results.params.values,
    'odds ratio': np.exp(results.params.values),
    'pvalue': results.pvalues,
    'name': results.params.index
})
coefs


# In[ ]:




