import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


''' https://stackabuse.com/decision-trees-in-python-with-scikit-learn/ '''


# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("train_test.csv")






train = df.iloc[:891]
test = df.iloc[891:]

# split into attributes and labels
target = 'Survived'
X_train = train.drop([target], axis=1)
y_train = train[target]

# Drop PassengerID for Training
X_train = X_train.drop(['PassengerId'], axis=1)
y_submit = test['PassengerId'].reset_index(drop=True)
X_test = test.drop(['PassengerId','Survived'], axis=1)

'''Printing Options'''
print("Xtrain")
print(X_train)
# print("ytrain")
# print(y_train)
print("Xtest")
print(X_test)
# print("ysubmit")
# print(y_submit)

# Convert features to Ordinal values
ordinalencoder_X = OrdinalEncoder()
X_train_1 = ordinalencoder_X.fit_transform(X_train)
X_train_1 = X_train_1.astype(int)

# Convert features to OneHotEncoding values
one_hot_encoder_X = OneHotEncoder()
X_train_2 = one_hot_encoder_X.fit_transform(X_train).toarray()
X_train_2 = X_train_2.astype(int)

# Convert target to Ordinal values
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)


cl_list = [RandomForestClassifier,
           MultinomialNB,
           BernoulliNB,
           SVC]

cl_trained_list = []

cl_strings = ['RandomForrest','MultinominalNB','BernoulliNB','SVC']

for i in range(len(cl_list)):
    # choose the classifier
    cl = cl_list[i]()
    scores = cross_val_score(cl, X_train_1, y_train, cv=10, scoring='accuracy')
    print("{} Ordinal Encoded".format(cl_strings[i]))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    cl_trained_list.append((scores.mean(), cl, 'oe'))

    scores = cross_val_score(cl, X_train_2, y_train, cv=10, scoring='accuracy')
    print("{} One Hot Encoded".format(cl_strings[i]))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    cl_trained_list.append((scores.mean(), cl, 'ohe'))

cl_trained_list = sorted(cl_trained_list,reverse=True)
best_cl = cl_trained_list[0]

print("The best classifier is: ",best_cl, "with encoding", cl_trained_list[0][2])

'''Fit the Classifier'''
if best_cl[2] == 'oe':
    train_cl = best_cl[1].fit(X_train_1, y_train)
elif best_cl[2] == 'ohe':
    train_cl = best_cl[1].fit(X_train_2, y_train)
else:
    print("\n\nError During Encoding\n\n")

'''Predict the test data'''
if best_cl[2] == 'oe':
    # Convert features to Ordinal values
    X_test = ordinalencoder_X.fit_transform(X_test)
    X_test = X_test.astype(int)
elif best_cl[2] == 'ohe':
    # Convert features to OneHotEncoding values
    one_hot_encoder_X = OneHotEncoder()
    X_test = one_hot_encoder_X.fit_transform(X_test).toarray()
    X_test = X_test.astype(int)
else:
    print("\n\nError During Encoding\n\n")

'''Join the results'''
pre_results = train_cl.predict(X_test)

print(pre_results)

results = pd.DataFrame({'Survived':pre_results})

print('results')
print(results)

df_submission = pd.concat([y_submit, results], axis=1)


print(df_submission)

# Create a csv for the merged datasets
df_submission.to_csv('submission2.csv', index=False)