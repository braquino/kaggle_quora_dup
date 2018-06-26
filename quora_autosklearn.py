from model import create_model
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

h = .02  # step size in the mesh

df_prov = []
for file in [x for x in os.listdir('data/train/')]:
    df_prov.append(pd.read_csv('data/train/' + file))
df = pd.concat(df_prov)

del df_prov

y = df['is_duplicate']
X = df[['match_words', 'match_sub_root', 'match_set', 'n_words', 'chunk_sim', 'n_verb', 'word_importance']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


names = ["SGD", "Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    SGDClassifier(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

for i, classifier in enumerate(classifiers):
    classifier.fit(X_train, y_train)
    print((names[i], classifier.score(X_test, y_test)))



