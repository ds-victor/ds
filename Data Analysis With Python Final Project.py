#!/usr/bin/env python
# coding: utf-8

# **Importing Required Libraries**

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ## Module 1: Importing Data Set

# In[2]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name, header = 0)
df.head()


# In[3]:


#Question 1:
df.dtypes


# In[6]:


df.describe(include = "all")


# # Module 2: Data Wrangling

# In[7]:


#Question 2
df.drop(["id", "Unnamed: 0"], axis = 1, inplace = True)
df.describe()


# In[8]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[12]:


#Replacing the NaN values in "bedrooms" column by the mean value
mean_bedrooms = df["bedrooms"].mean()
print("mean bedrooms is:", mean_bedrooms)
df["bedrooms"].replace(np.nan, mean_bedrooms, inplace = True)


# In[13]:


#Replacing the NaN values in "bathrooms" column by the mean value
mean_bathrooms = df["bathrooms"].mean()
print("mean bathrooms is:", mean_bathrooms)
df["bathrooms"].replace(np.nan, mean_bedrooms, inplace = True)


# In[14]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[31]:


#Question 3
df["floors"].value_counts().to_frame()


# In[51]:


#Question 4: 
#Converting the numeric "waterfront" column to a new categoric column
df["waterfront_type"] = df["waterfront"].apply(lambda x: "Without_Waterfront" if x == 0 else "With_Waterfront" )

#Plotting the box plot
plt.figure(figsize = (10,5))
sns.boxplot(x = df["waterfront_type"], y = df["price"], hue = df["waterfront_type"])
plt.title("Prce Vs Waterfront View")


# In[52]:


#Question 5
plt.figure(figsize = (10,6))
sns.regplot(x = "sqft_above", y = "price", data = df)
plt.title("sqft_above vs price Regplot")


# In[59]:


df.corr()['price'].sort_values()


# # Module 4: Model Development

# In[60]:


X = df[['long']]
Y = df['price']
lr = LinearRegression()
lr.fit(X,Y)
lr.score(X, Y)


# In[67]:


# Question 6
lr1 = LinearRegression()
lr1.fit(df[["sqft_living"]], df["price"])
predicted_price = lr1.predict(df[["sqft_living"]])
print("The predicted price are:",  predicted_price[0:5])
print()
print("The R^2 value is:", lr1.score(df[["sqft_living"]], df["price"]))


# In[69]:


# Question 7
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
mlr = LinearRegression()
mlr.fit(df[features], df["price"])
predicted_price_mlr = mlr.predict(df[features])
print("The predicted price for MLR model are:",  predicted_price_mlr[0:5])
print()
print("The R^2 value for MLR Model is:", mlr.score(df[features], df["price"]))


# In[71]:


# Question 8
#Creating the Pipeline object
Z = df[features]
Y = df["price"]
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe_obj = Pipeline(Input)

# Fitting the model
pipe_obj.fit(Z, Y)
predicted_price_pipe = pipe_obj.predict(Z)
print("The predicted prices are: ", predicted_price_pipe[0:5])


# In[72]:


# Calculating the R^2
Rsqr = pipe_obj.score(Z, Y)
print("The R^2 value is: ", Rsqr)


# # Module 5: Model Evaluation and Refinement

# In[73]:


# Importing the necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[76]:


# Splitting the data into train & test set
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
x = df[features]
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 1)

print("number of test samples:", x_test.shape[0])
print()
print("number of training samples:", x_train.shape[0])


# In[78]:


# Question 9: Ridge Regression with alpha = 0.1
# Importing the module
from sklearn.linear_model import Ridge


# In[80]:


# Creating the ridge object & fitting the model
RR = Ridge(alpha = 0.1)
RR.fit(x_train, y_train)
predicted_price_RR = RR.predict(x_test)
print("The Predicted price are: ", predicted_price_RR[0:5])


# In[81]:


# Calculating the R^2
Rsqr = RR.score(x_test, y_test)
print("The value of R^2 is: ", Rsqr)


# In[90]:


# Question 10: Ridge Regression with Polynomial Transformation with degree = 2
# Transforming the data
pr = PolynomialFeatures(degree = 2)
x_pr = pr.fit_transform(x)
x_train_pr, x_test_pr, y_train_pr, y_test_pr = train_test_split(x_pr, y, test_size = 0.2, random_state = 1)


# In[91]:


# Creating the ridge object & fitting the model
RR = Ridge(alpha = 0.1)
RR.fit(x_train_pr, y_train_pr)
predicted_price_PR = RR.predict(x_test_pr)
print("The Predicted price are: ", predicted_price_RR[0:5])


# In[92]:


# Calculating the R^2
Rsqr = RR.score(x_test_pr, y_test)
print("The value of R^2 is: ", Rsqr)


# In[ ]:




