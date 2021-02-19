import pandas as pd
import numpy as np

'''https://www.kaggle.com/blurredmachine/titanic-survival-a-complete-guide-for-beginners'''


# show complete records by changing rules
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)




# Load the dataset into the dataframe and combine
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')
df = pd.concat([df_train,df_test])


# # extracting titles from Name column.
# df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
#
# df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Other')
# df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
# df['Title'] = df['Title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'Dr'], 'Mrs')
# df['Title'] = df['Title'].replace(['Don'], 'Mr')
#
#
# Update attributes with missing Values with a random value based around the mean
null_attribute_list_numeric = [('LotFrontage',5), ('GarageYrBlt',5)]

for (attribute,split) in null_attribute_list_numeric:
    attr_avg = df[attribute].mean()
    attr_std = df[attribute].std()
    attr_null_count = df[attribute].isnull().sum()
    attr_null_random_list = np.random.randint(attr_avg - attr_std, attr_avg + attr_std, size=attr_null_count)
    df.loc[df[attribute].isnull(), attribute] = attr_null_random_list
    df[attribute] = df[attribute].astype(int)

    # Split Attribute into brackets
    att_string = attribute + 'band'
    df[att_string] = pd.cut(df[attribute], split)

    # Drop old attribute
    df.drop([attribute], axis=1, inplace=True)

# # These features are missing and will be imputed with zero
zero_attribute_list_categoric = ['GarageCars', 'GarageArea', ]

for attribute in zero_attribute_list_categoric:
    df.loc[df[attribute].isnull(), attribute] = 0

# # These features are not applicable and should not be imputed with a missing value
na_attribute_list_categoric = ['Alley', 'Exterior1st', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                               'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                               'PoolQC', 'Fence', 'MiscFeature']

for attribute in na_attribute_list_categoric:
    df.loc[df[attribute].isnull(), attribute] = "Not Applicable"

# These features are missing and will be imputed with the largest category from the feature
missing_attribute_list_categoric = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                                    'MasVnrType', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                                    'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
                                    'KitchenQual', 'Functional', 'GarageQual', 'GarageCond', 'SaleType']

for attribute in missing_attribute_list_categoric:
    df[attribute].fillna(df[attribute].mode()[0], inplace=True)


# Convert numeric values into smaller bands
numeric_attributes_into_bands = [('LotArea', 5), ('YearBuilt', 10), ('YearRemodAdd', 10),
                                 ('MasVnrArea', 5), ('BsmtFinSF1', 5), ('BsmtFinSF2', 5), ('BsmtUnfSF', 5),
                                 ('TotalBsmtSF', 5), ('1stFlrSF', 5), ('2ndFlrSF', 5), ('LowQualFinSF', 5),
                                 ('GrLivArea', 5), ('GarageArea', 5), ('WoodDeckSF', 5),
                                 ('OpenPorchSF', 5), ('EnclosedPorch', 5), ('3SsnPorch', 3), ('ScreenPorch', 4),
                                 ('MiscVal', 12)
                                 ]

for (attribute,split) in numeric_attributes_into_bands:
    att_string = attribute + 'band'
    df[att_string] = pd.cut(df[attribute], split)
    df.drop([attribute], axis=1, inplace=True)


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

#
#



'''Final Checks - Printing Values'''
print(df.head(20))
# print(pd.crosstab(df['CreatineBand'], df['DEATH_EVENT']))
print("Display Nulls")
print(df.isnull().sum(axis = 0))
print("Print rows where BsmtExposure is Null")
print(df[df['BsmtExposure'].isnull()])
print("\n\nDescription of BsmtCond")
print(df.BsmtCond.describe())
# print("\n\nDescription of Fare")
# print(df.Fare.describe())
print("Display the number of unique values for each column")
print(df.nunique())

# print(df.dtypes)

# Printing different value counts
print_list = []
for feature in print_list:
    print("Printing information about: ", feature)
    print(df[feature].value_counts())





'''Plotting'''
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(x = df['Age'], y = df['Fare'])
# plt.xlabel("Age")
# plt.ylabel("Fare")
#
# plt.show()


# Create a csv for the merged datasets
df.to_csv('PreProcessed/HouseProces_pp2', index=False)
