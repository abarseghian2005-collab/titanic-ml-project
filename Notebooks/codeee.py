import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import joblib
scaler=StandardScaler()
df = pd.read_csv('../data/train_data.csv')
dff=pd.read_csv('../data/test_data.csv')
x_train=df.drop('Survived', axis=1)
y_train=df['Survived']
x_test=dff.drop('Survived',axis=1)
y_test=dff['Survived']
x_train_scl=scaler.fit_transform(x_train)
x_test_scl=scaler.transform(x_test)
nmodel=GaussianNB()
logmodel=LogisticRegression(max_iter=10000)
kmodel=KNeighborsClassifier(n_neighbors=10)
tmodel=DecisionTreeClassifier(max_depth = 3)
ldamodel, qdamodel=LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(reg_param=0.05)
ldamodel.fit(x_train_scl,y_train)
nmodel.fit(x_train_scl,y_train)
qdamodel.fit(x_train_scl,y_train)
logmodel.fit(x_train_scl, y_train)
kmodel.fit(x_train_scl,y_train)
tmodel.fit(x_train_scl,y_train)
logpr=logmodel.predict(x_test_scl)
kpr=kmodel.predict(x_test_scl)
npr=nmodel.predict(x_test_scl)
tpr=tmodel.predict(x_test_scl)
ldapr=ldamodel.predict(x_test_scl)
qdapr=qdamodel.predict(x_test_scl)
print("results for logistic model", accuracy_score(y_test,logpr),'\n', confusion_matrix(y_test,logpr))
print("results for kneighboor model", accuracy_score(y_test,kpr),'\n', confusion_matrix(y_test,kpr))
print("results for tree model", accuracy_score(y_test,tpr),'\n', confusion_matrix(y_test,tpr))
print("results for lda model", accuracy_score(y_test,ldapr),'\n', confusion_matrix(y_test,ldapr))
print("results for qda model", accuracy_score(y_test,qdapr),'\n', confusion_matrix(y_test,qdapr))
print("results for naive bayes model", accuracy_score(y_test,npr),'\n', confusion_matrix(y_test,npr))
joblib.dump(nmodel, '../models/GaussianNB.pkl')
joblib.dump(logmodel, '../models/LogisticRegression.pkl')
joblib.dump(kmodel, '../models/KNeighborsClassifier.pkl')
joblib.dump(tmodel, '../models/DecisionTreeClassifier.pkl')
joblib.dump(ldamodel, '../models/LinearDiscriminantAnalysis.pkl')
joblib.dump(qdamodel, '../models/QuadraticDiscriminantAnalysis.pkl')