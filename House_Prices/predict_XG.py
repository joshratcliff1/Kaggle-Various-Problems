import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

''' https://stackabuse.com/decision-trees-in-python-with-scikit-learn/ '''

# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("PreProcessed/HousePrices_simple_pp.csv")

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


encoding = 'ordinal'
if encoding == 'ordinal':
    # Convert categorical features (datatype=object) to Ordinal values
    objList = X.select_dtypes(include="object").columns

    # Label Encoding for object to ordinal conversion
    le = LabelEncoder()
    for feat in objList:
        X[feat] = le.fit_transform(X[feat].astype(str))

elif encoding == 'ohe':
    # Convert features to OneHotEncoding values
    one_hot_encoder_X = OneHotEncoder()
    X = one_hot_encoder_X.fit_transform(X).toarray()
    X = X.astype(int)

# Create the train and test data for the two encodings
X_train_full = X[:train_quant]
X_submit = X[train_quant:]

# Create train and test samples
train_X, test_X, train_y, test_y = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=123)

predictor1 = XGBRegressor()

''' https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f '''

# Create a DataMatrices (Datastructure native to XGBoost), for testing, training and submitting
dtrain = xgb.DMatrix(data=train_X, label=train_y)
dtest = xgb.DMatrix(data=test_X, label=test_y)
dsubmit = xgb.DMatrix(data=X_submit, label=y_submit)


# Adding cross validation
params = {"objective": "reg:squarederror",
          'learning_rate': 0.1,
          'max_depth': 9,
          'min_child_weight': 7,
          'eta':0.01,
          'subsample': 0.9,
          'colsample_bytree': 1,
          'alpha': 10}

''' Uncomment below to perform tuning of parameters '''
tune_flag = True

if tune_flag:
    gridsearch_params1 = [
        (max_depth, min_child_weight)
        for max_depth in range(9,15)
        for min_child_weight in range(3,10)]

    gridsearch_params2 = [
        (subsample, colsample_bytree)
        for subsample in [i/10. for i in range(7,11)]
        for colsample_bytree in [i/10. for i in range(7,11)]]

    all_gridsearch_params = [(gridsearch_params1, 'max_depth', 'min_child_weight'),
                             (gridsearch_params2, 'subsample', 'colsample_bytree')]

    # Performing Grid Search
    for gridsearch_params, param1_string, param2_string in all_gridsearch_params:
        min_rmse = float("Inf")
        best_params = None
        for param1, param2 in gridsearch_params:
            print("CV with max_depth={}, min_child_weight={}".format(
                                     param1,
                                     param2))
            # Update our parameters
            params[param1_string] = param1
            params[param2_string] = param2
            # Run CV
            cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=5,
                                num_boost_round=999, early_stopping_rounds=10,
                                metrics="rmse", as_pandas=True, seed=123)
            # Update best RMSE
            mean_rmse = cv_results['test-rmse-mean'].min()
            boost_rounds = cv_results['test-rmse-mean'].argmin()
            print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = (param1, param2)
        print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))

        # Update Parameters with optimal
        params[param1_string] = best_params[0]
        params[param2_string] = best_params[1]

trained_pr = xgb.train(
    params,
    dtrain,
    num_boost_round=999,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

print("Best RMSE: {:.5f} in {} rounds".format(trained_pr.best_score, trained_pr.best_iteration+1))

best_pr = xgb.train(
    params,
    dtrain,
    num_boost_round=trained_pr.best_iteration+1,
    evals=[(dtest, "Test")]
)

# Join the results
pre_results = best_pr.predict(dsubmit)
results = pd.DataFrame({'SalePrice': pre_results})
df_submission = pd.concat([y_submit, results], axis=1)

# Perform an Exponential Transformation. This will return the values to the correct scale.
df_submission['SalePrice'] = np.expm1(df_submission['SalePrice'])

# Create a csv for the merged datasets
df_submission.to_csv('Submit/submission_XG_tuned_simple.csv', index=False)
