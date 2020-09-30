from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
            
class Classifier:
    def __init__(self, name, X, y):
      self.name = name
      self.X_train = X
      self.y_train = y
      self.fact = 0.32

    def classify(self, X_test):
        if self.name == 'KNN':
            n = KNeighborsClassifier(n_neighbors=4)
            n.fit(self.X_train, self.y_train)
            return n.predict(X_test)
            
        elif self.name == 'Decision Tree':
            decision_tree = DecisionTreeClassifier(max_depth=3)
            decision_tree.fit(self.X_train, self.y_train)
            return decision_tree.predict(X_test)
        
        elif self.name == 'Naive Bayes':
            b = GaussianNB()
            b.fit(self.X_train, self.y_train)
            return b.predict(X_test)
            
        elif self.name == "Random Forest":
            r = RandomForestClassifier(n_estimators=200, max_depth = 3)
            r.fit(self.X_train, self.y_train)
            return r.predict(X_test)

    def calculateAccuracy(self, y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred) * 100
    
        
        
    
    
        
        
