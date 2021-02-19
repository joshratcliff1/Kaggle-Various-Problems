import pandas as pd
import numpy as np

'''https://www.kaggle.com/blurredmachine/titanic-survival-a-complete-guide-for-beginners'''


# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)




# Load the dataset into the dataframe and combine
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# # extracting titles from Name column.
# df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
#
# df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Other')
# df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
# df['Title'] = df['Title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'Dr'], 'Mrs')
# df['Title'] = df['Title'].replace(['Don'], 'Mr')
#
#
# # Update missing Age Values with a random age
# age_avg = df['Age'].mean()
# age_std = df['Age'].std()
# age_null_count = df['Age'].isnull().sum()
# age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# df.loc[df['Age'].isnull(), 'Age'] = age_null_random_list
# df['Age'] = df['Age'].astype(int)
#
# # Split Age into brackets
# df['AgeBand'] = pd.cut(df['Age'], 10)
#
# # Update missing Fare Values with a random fare
# fare_avg = df['Fare'].mean()
# fare_std = df['Fare'].std()
# fare_null_count = df['Fare'].isnull().sum()
# fare_null_random_list = np.random.randint(fare_avg - fare_std, fare_avg + fare_std, size=fare_null_count)
# df.loc[df['Fare'].isnull(), 'Fare'] = fare_null_random_list
#
# # Split Fare into brackets
# df['FareBand'] = pd.cut(df['Fare'], 10)
#
#
# # Create a family attribute
# df['Family'] = df['SibSp'] + df['Parch'] + 1
#
# # Create an Is Alone Attribute
# df['MaleIsAlone'] = 0
# df['FemaleIsAlone'] = 0
# df.loc[(df['Family'] == 1) & (df['Sex'] == 'male'), 'MaleIsAlone'] = 1
# df.loc[(df['Family'] == 1) & (df['Sex'] == 'female'), 'FemaleIsAlone'] = 1
#
# # Replace other NaN Values
# df.Cabin = df.Cabin.fillna('Missing')
# df.Embarked = df.Embarked.fillna('S')
#
# # Create a Fare/Age Attribute
# df['Fare_Age'] = df['Fare'] / df['Age']
# df['Fare_Age_Band'] = pd.qcut(df['Fare'], 6)
#
#
# # Delete unnecessary attributes
# df.drop(['Name'], axis=1, inplace=True)
# df.drop(['Age'], axis=1, inplace=True)
# df.drop(['Fare'], axis=1, inplace=True)
# df.drop(['Ticket'], axis=1, inplace=True)
# df.drop(['Cabin'], axis=1, inplace=True)
# df.drop(['SibSp'], axis=1, inplace=True)
# df.drop(['Parch'], axis=1, inplace=True)
# df.drop(['Fare_Age'], axis=1, inplace=True)



# Convert to bands
df['AgeBand'] = pd.qcut(df['age'], 5)
df['CreatineBand'] = pd.qcut(df['creatinine_phosphokinase'], 5)
df['EFBand'] = pd.qcut(df['ejection_fraction'], 5)
df['PlateletsBand'] = pd.qcut(df['platelets'], 5)
df['Serum_CreatinineBand'] = pd.qcut(df['serum_creatinine'], 5)
df['Serum_SodiumBand'] = pd.qcut(df['serum_sodium'], 5)
df['TimeBand'] = pd.qcut(df['time'], 5)

df.drop(['age'], axis=1, inplace=True)
df.drop(['creatinine_phosphokinase'], axis=1, inplace=True)
df.drop(['ejection_fraction'], axis=1, inplace=True)
df.drop(['platelets'], axis=1, inplace=True)
df.drop(['serum_creatinine'], axis=1, inplace=True)
df.drop(['serum_sodium'], axis=1, inplace=True)
df.drop(['time'], axis=1, inplace=True)


'''Final Checks - Printing Values'''
print(df.head())
print(pd.crosstab(df['CreatineBand'], df['DEATH_EVENT']))
# print("Display Nulls")
# print(df.isnull().sum(axis = 0))
# print("Print rows where Age is Null")
# print(df[df['Age'].isnull()])
# print("\n\nDescription of Age")
# print(df.Age.describe())
# print("\n\nDescription of Fare")
# print(df.Fare.describe())

# print(df.dtypes)
# print(df['age_gender'].value_counts())


'''Plotting'''
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(x = df['Age'], y = df['Fare'])
# plt.xlabel("Age")
# plt.ylabel("Fare")
#
# plt.show()


# Create a csv for the merged datasets
df.to_csv('heart_failure_pp.csv', index=False)
