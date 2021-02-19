import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn import linear_model
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier

''' https://stackabuse.com/decision-trees-in-python-with-scikit-learn/ '''

# show complete records by changing rules
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("PreProcessed/HousePrices_FE.csv")

# split into attributes and labels
target = 'SalePrice'
X = df.copy().drop([target], axis=1)
Xcols = X.columns
y = df.copy()[[target, 'Id']]

# Perform Log Transformation. Will perform exponential transformation at end to data.
y[target] = np.log1p(y[target])

# Create ytrain and ysubmit
train_quant = 1460
y_train_full = y.iloc[:train_quant]
y_train_full = y_train_full[target]
y_submit = y.iloc[train_quant:]
y_submit = y_submit['Id'].reset_index(drop=True)

# Convert categorical features (datatype=object) to Ordinal values
objList = X.select_dtypes(include="object").columns

# Label Encoding for objects (categorical features) to ordinal conversion
le = LabelEncoder()
for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))


# Create the train and test data for the two encodings
X_train_full = X.copy()[:train_quant]
X_submit = X.copy()[train_quant:]

'''This section was trying to pull out important features'''
# Feature Extraction
xgb = XGBRegressor()
xgb.fit(X_train_full, y_train_full)
df_imp = pd.DataFrame(xgb.feature_importances_, columns=['Importance'],index=Xcols)
df_imp = df_imp.sort_values(['Importance'], ascending=False)

print(df_imp)
low_imp_cols = df_imp.index[-34:].tolist()
print(low_imp_cols)


for column in low_imp_cols:
    X_train_full.drop([column], axis=1, inplace=True)
    X_submit.drop([column], axis=1, inplace=True)







# Create the predictors in a dictionary
predictors = {'linear_regression': LinearRegression(),
              'Random Forest': RandomForestRegressor(),
              'linear_model': linear_model.Lasso(alpha=0.1),
              #'XGBoost': XGBRegressor(n_estimators=1000, learning_rate=0.05),
              #'XGBoost-SK': GradientBoostingRegressor(),
              'XGBoost-SK-Params': GradientBoostingRegressor(learning_rate=0.05, n_estimators=500, max_features='sqrt',
                                                       max_depth=4)
}

pr_trained_list = []

for name, predictor in predictors.items():
    scores = cross_val_score(predictor, X_train_full, y_train_full, cv=5, scoring='neg_root_mean_squared_error')
    print(name)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    pr_trained_list.append((scores.mean(), predictor, name))

pr_trained_list.sort(key=lambda x: -x[0])
best_pr = pr_trained_list[0]

print("The best predictor is: ", best_pr[2], 'with score of', best_pr[0])

# Fit the Predictor
if best_pr[2] == 'XGBoost': # Different requirments for the XGBoost model
    train_pr = best_pr[1].fit(X_train_full, y_train_full, early_stopping_rounds=10,
                              eval_set=[(X_submit, y_submit)], verbose=False)
else:
    train_pr = best_pr[1].fit(X_train_full, y_train_full)

# Join the results
pre_results = train_pr.predict(X_submit)
results = pd.DataFrame({'SalePrice': pre_results})
df_submission = pd.concat([y_submit, results], axis=1)

# Perform an Exponential Transformation. This will return the values to the correct scale.
df_submission['SalePrice'] = np.expm1(df_submission['SalePrice'])

# Create a csv for the merged datasets
import datetime
today = datetime.datetime.now()
date_time = today.strftime("%Y%m%d_%H_%M_%S")
df_submission.to_csv(f'Submit/submission_{date_time}.csv', index=False)
