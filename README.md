<H3>ENTER YOUR NAME: SAI GURUCHANDRAN G</H3>
<H3>ENTER YOUR REGISTER NO: 212223240143</H3>
<H3>EX. NO.1</H3>
<H3>DATE 21.04.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# import libraries
from google.colab import files
import pandas as pd
import io

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Read the dataset
df = pd.read_csv('Churn_Modelling.csv')
print(df.head())

# Finding Missing Values
print(df.isnull().sum())

# Handling Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Handle categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split dataset into input and output
X = df.drop('Exited', axis=1)
y = df['Exited']

print(X.head())
print(y.head())
#  Check for Duplicates
print(df.duplicated().sum())

#  Remove Duplicates (optional)
df.drop_duplicates(inplace=True)
#  Detect Outliers (use existing numeric columns)
print(df.describe())
print(df['Balance'].describe())
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1.head())
#  Splitting the data for training & testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 'test_size=0.2' means 20% test data and 80% training data

#  Print results
print("X_train:\n", X_train.head())
print("Length of X_train:", len(X_train))

print("\nX_test:\n", X_test.head())
print("Length of X_test:", len(X_test))
```


## OUTPUT:
<img width="970" height="688" alt="image" src="https://github.com/user-attachments/assets/4c463d00-7006-477f-9252-90935b9670a2" />
<img width="993" height="751" alt="image" src="https://github.com/user-attachments/assets/1c7ab21d-81a7-4f9b-a274-51672e26aeba" />
<img width="979" height="576" alt="image" src="https://github.com/user-attachments/assets/c6a88be9-fef1-42df-929b-7dd75e1878ef" />
<img width="451" height="185" alt="image" src="https://github.com/user-attachments/assets/cdf6dee0-d378-4e68-aba7-ca57673576a3" />
<img width="832" height="302" alt="image" src="https://github.com/user-attachments/assets/05655296-f8a3-4b8d-9645-a3657dcb3368" />
<img width="963" height="692" alt="image" src="https://github.com/user-attachments/assets/36d214ac-278e-41a4-922d-0380fe1f387e" />
<img width="782" height="750" alt="image" src="https://github.com/user-attachments/assets/17ed068a-ff9c-4df4-87c8-3a6c63004946" />







## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


