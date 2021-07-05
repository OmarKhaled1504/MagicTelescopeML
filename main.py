import random
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def split(instances):
    x = instances.values[:, 0:9]
    y = instances.values[:, 10]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100)
    return x, y, x_train, x_test, y_train, y_test


def training_with_gini(x_train, x_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
    clf_gini.fit(x_train, y_train)
    return clf_gini


def training_with_entropy(x_train, x_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100)
    clf_entropy.fit(x_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


def decision_tree(x_train, x_test, y_train, y_test):
    clf_gini = training_with_gini(x_train, x_test, y_train)
    clf_entropy = training_with_entropy(x_train, x_test, y_train)
    y_pred_gini = prediction(x_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    y_pred_entropy = prediction(x_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__ == "__main__":
    instances = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data',
        sep=',', header=None)

    i = 5644  # 5644 is the number of classes (g) that should be randomly removed for the two classes to be balanced
    while i > 0:
        j = random.randint(0, 12331)
        if instances.iloc[j, 10] == 'g':
            instances = instances.drop(instances.index[j])
            i -= 1
    x, y, x_train, x_test, y_train, y_test = split(instances)
    decision_tree(x_train, x_test, y_train, y_test)

    # print(len(instances))
