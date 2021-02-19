import pandas as pd
import numpy as np
from xgboost import XGBRegressor

'''https://www.kaggle.com/blurredmachine/titanic-survival-a-complete-guide-for-beginners'''


# show complete records by changing rules
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)




# Load the dataset into the dataframe and combine
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')
df = pd.concat([df_train,df_test])

# Innitial peak
print(df.head(20))

# Update attributes with missing Values with a random value based around the mean
null_attribute_list_numeric = [('LotFrontage',5), ('GarageYrBlt',5)]

for (attribute,split) in null_attribute_list_numeric:
    attr_avg = df[attribute].mean()
    attr_std = df[attribute].std()
    attr_null_count = df[attribute].isnull().sum()
    attr_null_random_list = np.random.randint(attr_avg - attr_std, attr_avg + attr_std, size=attr_null_count)
    df.loc[df[attribute].isnull(), attribute] = attr_null_random_list
    df[attribute] = df[attribute].astype(int)

    # # Split Attribute into brackets
    # att_string = attribute + 'band'
    # df[att_string] = pd.cut(df[attribute], split)
    #
    # # Drop old attribute
    # df.drop([attribute], axis=1, inplace=True)

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

# Create a more accurate Year Sold feature
df['YrSold_precise'] = df['YrSold'] + (df['MoSold'] / 12)
df.drop(['YrSold'], axis=1, inplace=True)

# Create a Quality multiplied by condition feature
df['QualCond'] = df['OverallQual'] * df['OverallCond']



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
print('\n\n')
print(df.head(20))
# print(pd.crosstab(df['CreatineBand'], df['DEATH_EVENT']))
# print("Display Nulls")
# print(df.isnull().sum(axis = 0))
# print("Print rows where BsmtExposure is Null")
# print(df[df['BsmtExposure'].isnull()])
# print("\n\nDescription of BsmtCond")
# print(df.BsmtCond.describe())
# print("\n\nDescription of Fare")
# print(df.Fare.describe())
# print("Display the number of unique values for each column")
# print(df.nunique())





# Printing different value counts
# print_list = []
# for feature in print_list:
#     print("Printing information about: ", feature)
#     print(df[feature].value_counts())





'''Plotting'''
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(x = df['Age'], y = df['Fare'])
# plt.xlabel("Age")
# plt.ylabel("Fare")
#
# plt.show()


# Create a csv for the merged datasets
df.to_csv('PreProcessed/HousePrices_FE.csv', index=False)