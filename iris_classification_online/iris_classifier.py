import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
import pickle

iris=load_iris()
X=iris.data
y=iris.target
target_names=iris.target_names
#print(iris)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

tree_clf=DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train,y_train)

y_pred=tree_clf.predict(X_test)


print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

val=tree_clf.predict([[10,23,5,2.5]])
print(val,target_names[val])

print(tree_clf.feature_importances_)

filename="finalized_model.sav"

#pickle.dump(tree_clf,open(filename,"wb"))

target_names=['setosa', 'versicolor', 'virginica']
loaded_model=pickle.load(open(filename,"rb"))

val=loaded_model.predict([[10,23,5,2.5]])

#print(val)
print(target_names[val[0]]) 

#return target_names[val[0]]
