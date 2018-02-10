# Alexander Litvintsev
# alexanderlitvintsev@mail.ru
# Titanic: Machine Learning from Disaster from kaggle.com
# https://www.kaggle.com/c/titanic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Logistic Regression (Логистическая регрессия)
# http://www.machinelearning.ru/wiki/index.php?title=Регрессия
from sklearn.linear_model import LogisticRegression

# Support Vector Machine (SVM) (Метод опорных векторов)
# http://www.machinelearning.ru/wiki/index.php?title=SVM
from sklearn.svm import SVC, LinearSVC

# k-nearest neighbors algorithm (Метод ближайших соседей)
# http://www.machinelearning.ru/wiki/index.php?title=Метод_ближайшего_соседа
from sklearn.neighbors import KNeighborsClassifier

# Decision Tree Classifier (Дерево принятия решений)
# https://ru.wikipedia.org/wiki/Дерево_принятия_решений
from sklearn.tree import DecisionTreeClassifier

# Random Forest (Cлучайный лес)
# https://ru.wikipedia.org/wiki/Random_forest
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes classifier (Наивный байесовский классификатор)
# http://www.machinelearning.ru/wiki/index.php?title=Байесовский_классификатор
from sklearn.naive_bayes import GaussianNB

# Perceptron (Персептрон)
# http://www.machinelearning.ru/wiki/index.php?title=Персептрон
from sklearn.linear_model import Perceptron

# Stochastic gradient descent (Стохастический градиентный спуск)
# https://en.wikipedia.org/wiki/Stochastic_gradient_descent
from sklearn.linear_model import SGDClassifier

sns.set()

# PassengerId — идентификатор пассажира
# Survival — выжил человек (1) или нет (0)
# Pclass — Класс билета:
# 1 - высокий
# 2 - средний
# 3 - низкий
# Name — имя пассажира
# Sex — пол пассажира
# Age — возраст
# SibSp — Количество родственников 2-го порядка (муж, жена, братья, сетры)
# Parch — Количество родственников 1-го порядка (мать, отец, дети)
# Ticket — номер билета
# Fare — цена билета
# Cabin — каюта
# Embarked — порт посадки
# C — Cherbourg
# Q — Queenstown
# S — Southampton

# Загрузка выборки
train = pd.read_csv("titanic_data/train.csv")
test = pd.read_csv("titanic_data/test.csv")

# Обзор набора данных
print(train.head())
print("Size = ",train.shape)
print(train.info())
print(train.isnull().sum())


# Взаимосвязь между признаками и выживаемостью / Relationship between Features and Survival
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print("Total: %i"%len(train))

# Взаимосвязь между Классом и выживаемостью / Pclass vs. Survival
train.Pclass.value_counts()
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
sns.barplot(x='Pclass', y='Survived', data=train)

# Взаимосвязь между Полом и выживаемостью / Sex vs. Survival
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
sns.barplot(x='Sex', y='Survived', data=train)

# Взаимосвязь между Классом, Полом и выживаемостью / Pclass & Sex vs. Survival
tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')

sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)

# Взаимосвязь между Классом, Полом, Портом и выживаемостью / Pclass, Sex & Embarked vs. Survival
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)

# Взаимосвязь между Портом и выживаемостью
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=train)

# Взаимосвязь между Количеством родственников 1 и выживаемостью / Parch vs. Survival
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()

sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar

# Взаимосвязь между Количеством родственников 2 и выживаемостью / SibSp vs. Survival
train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()

sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar

# Взаимосвязь между Возрастом и выживаемостью / Age vs. Survival

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)



total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')

# Поиск взаимосвязей между признаками / Correlating Features

plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)

# Извлечение признаков / Feature Extraction
# Имя / Name Feature

# Объединение тренировочной и тестовой выборки
train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

train.head()
pd.crosstab(train['Title'], train['Sex'])

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

# Пол / Sex Feature

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()

# Порт посадки / Embarked Feature
train.Embarked.unique()
train.Embarked.value_counts()
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head()

for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()

# Возраст / Age Feature

for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

train.head()

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train.head()

# Цена билета / Fare Feature

for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

train.head()


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train.head()

# Количество родственников / SibSp & Parch Feature

for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train.head(1)

test.head(1)

# Выбор признаков / Feature Selection


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

train.head()

test.head()

# Классификация и точность / Classification & Accuracy

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape

# Logistic Regression (Логистическая регрессия)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')

# Support Vector Machine (SVM) (Метод опорных векторов)

clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)

# Linear SVM

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)

# k-Nearest Neighbors (Метод ближайших соседей)

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)

# Decision Tree (Дерево принятия решений)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)

# Random Forest (Cлучайный лес)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)

# Gaussian Naive Bayes (Наивный байесовский классификатор)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)

# Perceptron (Персептрон)


clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)


# Stochastic Gradient Descent (SGD) (Стохастический градиентный спуск)

clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)


# Матрица ошибок / Confusion Matrix

from sklearn.metrics import confusion_matrix

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix,
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent,
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

# Сравнение моделей / Comparing Models

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC',
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes',
              'Perceptron', 'Stochastic Gradient Decent'],

    'Score': [acc_log_reg, acc_svc, acc_linear_svc,
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb,
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)

test.head()

# Создание итогового файла / Create Submission File
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_random_forest
    })
submission.to_csv('submission.csv', index=False)

plt.show()
