import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Dosya yollarýný belirle
train_benign_file = r"C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_features\\train_benign_features.xlsx"
test_benign_file = r"C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_features\\test_benign_features.xlsx"
train_malignant_file = r"C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_features\\train_malignant_features.xlsx"
test_malignant_file = r"C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_features\\test_malignant_features.xlsx"

# Veri setlerini yükle
train_benign = pd.read_excel(train_benign_file)
test_benign = pd.read_excel(test_benign_file)
train_malignant = pd.read_excel(train_malignant_file)
test_malignant = pd.read_excel(test_malignant_file)

# Ýlk sütunu çýkar
train_benign = train_benign.iloc[:, 1:]
test_benign = test_benign.iloc[:, 1:]
train_malignant = train_malignant.iloc[:, 1:]
test_malignant = test_malignant.iloc[:, 1:]

# NaN deðerleri ortalama ile doldur
train_benign = train_benign.fillna(train_benign.mean())
test_benign = test_benign.fillna(test_benign.mean())
train_malignant = train_malignant.fillna(train_malignant.mean())
test_malignant = test_malignant.fillna(test_malignant.mean())

# Veri setlerini birleþtir
train_data = pd.concat([train_benign, train_malignant])
test_data = pd.concat([test_benign, test_malignant])

# Etiketleri oluþtur
train_labels = np.concatenate([np.zeros(len(train_benign)), np.ones(len(train_malignant))])
test_labels = np.concatenate([np.zeros(len(test_benign)), np.ones(len(test_malignant))])

# Min-Max scaling
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Feature selection için Chi-Square kullan
selector_chi2 = SelectKBest(score_func=chi2, k=5)  # Top 5 feature seçilecek
train_selected_chi2 = selector_chi2.fit_transform(train_data_scaled, train_labels)
test_selected_chi2 = selector_chi2.transform(test_data_scaled)

# Feature selection için Mutual Information kullan
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)  # Top 5 feature seçilecek
train_selected_mi = selector_mi.fit_transform(train_data_scaled, train_labels)
test_selected_mi = selector_mi.transform(test_data_scaled)

# Feature selection için RFE kullan
estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
selector_rfe = RFE(estimator, n_features_to_select=5)
train_selected_rfe = selector_rfe.fit_transform(train_data_scaled, train_labels)
test_selected_rfe = selector_rfe.transform(test_data_scaled)

# Feature selection için L1-based feature selection kullan
selector_l1 = SelectKBest(score_func=chi2, k=5)
train_selected_l1 = selector_l1.fit_transform(train_data_scaled, train_labels)
test_selected_l1 = selector_l1.transform(test_data_scaled)

# Machine learning algoritmalarýný eðit ve test et
classifiers = [
    RandomForestClassifier(),
    SVC(),
    XGBClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(),  # GBM
    DecisionTreeClassifier(),  # Decision Tree
    GaussianNB(),  # Naive Bayes
    LogisticRegression()  # Logistic Regression
]


feature_sets = [
    (train_selected_chi2, test_selected_chi2, "Chi-Square"),
    (train_selected_mi, test_selected_mi, "Mutual Information"),
    (train_selected_rfe, test_selected_rfe, "RFE"),
    (train_selected_l1, test_selected_l1, "L1-based")
]

best_accuracy = 0
best_method = None
best_features = None
best_feature_names = None
best_classifier = None

for features, test_features, method_name in feature_sets:
    print(f" ")
    print(f"--Feature Selection Method: {method_name}")
    for clf in classifiers:
        clf.fit(features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy for {clf.__class__.__name__}: {accuracy}")

        # Check if the current accuracy is better than the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method_name
            best_features = features
            best_feature_names = train_data.columns[selector_rfe.support_]
            best_classifier = clf

print("-------------------------------")
print("Best Results:")
print(f"Best Feature Selection Method: {best_method}")
print(f"Best Classifier: {best_classifier.__class__.__name__}")
print(f"Best Accuracy: {best_accuracy}")
print("Selected Features:")
print(best_feature_names)
