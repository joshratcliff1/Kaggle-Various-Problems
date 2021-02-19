import pandas as pd
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

''' https://stackabuse.com/decision-trees-in-python-with-scikit-learn/ '''


# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("train_test.csv")

# split into attributes and labels
target = 'Survived'
X = df.drop([target], axis=1)
y = df[[target, 'PassengerId']]

# Create ytrain and ysubmit
y_train = y.iloc[:891]
y_train = y_train[target]
y_submit = y.iloc[891:]
y_submit = y_submit['PassengerId'].reset_index(drop=True)

# Convert features to Ordinal values
ordinalencoder_X = OrdinalEncoder()
X_1 = ordinalencoder_X.fit_transform(X)
X_1 = X_1.astype(int)

# Convert features to OneHotEncoding values
one_hot_encoder_X = OneHotEncoder()
X_2 = one_hot_encoder_X.fit_transform(X).toarray()
X_2 = X_2.astype(int)

# Convert target to Ordinal values
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)


# Create the train and test data for the two encodings
# print(X_1)
X_train_1 = X_1[:891]
X_train_2 = X_2[:891]

X_test_1 = X_1[891:]
X_test_2 = X_2[891:]

'''Printing Options'''
# print("Xtrain_1")
# print(X_train_1)
# print("ytrain")
# print(y_train)
# print("Xtest")
# print(X_test)
# print("ysubmit")
# print(y_submit)


# Create the classifiers
cl_list = [RandomForestClassifier(),
           MultinomialNB(),
           BernoulliNB(),
           GaussianNB(),
           SVC(),
           #GradientBoostingClassifier(),
           AdaBoostClassifier(),
           KNeighborsClassifier(n_neighbors = 3),
           #DecisionTreeClassifier(),
           Perceptron(max_iter=5, tol=None),
           SGDClassifier(max_iter=5, tol=None)]

cl_trained_list = []

cl_strings = ['RandomForrest',
              'MultinominalNB',
              'BernoulliNB',
              'GaussianNB',
              'SVC',
              #'GradientBoostingClassifier',
              'AdaBoost',
              'K Nearest Neighbours',
              #'Decision Tree',
              'Perceptron',
              'Stochastic Gradient Descent']

for i in range(len(cl_list)):
    # choose the classifier
    cl = cl_list[i]
    scores = cross_val_score(cl, X_train_1, y_train, cv=10, scoring='accuracy')
    print("{} Ordinal Encoded".format(cl_strings[i]))
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    cl_trained_list.append((scores.mean(), cl, 'oe'))

    scores = cross_val_score(cl, X_train_2, y_train, cv=10, scoring='accuracy')
    print("{} One Hot Encoded".format(cl_strings[i]))
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    cl_trained_list.append((scores.mean(), cl, 'ohe'))

print(cl_trained_list)
# cl_trained_list = sorted(cl_trained_list[0],reverse=True)
cl_trained_list.sort(key=lambda x: -x[0])
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
    X_test = X_test_1
    X_test = X_test.astype(int)
elif best_cl[2] == 'ohe':
    X_test = X_test_2
    X_test = X_test.astype(int)
else:
    print("\n\nError During Encoding\n\n")

# Join the results
pre_results = train_cl.predict(X_test)
results = pd.DataFrame({'Survived':pre_results})
df_submission = pd.concat([y_submit, results], axis=1)

# Create a csv for the merged datasets
df_submission.to_csv('submission8.csv', index=False)