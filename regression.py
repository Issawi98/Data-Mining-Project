from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

class Regression:
    def __init__(self, regressor_name, X, y):
      self.regressor_name = regressor_name
      self.X_train = X
      self.y_train = y
    
    def predict(self, X_test):
        if self.regressor_name == 'Decision Tree':
            decision_tree = DecisionTreeRegressor(random_state=0, max_depth=2)
            decision_tree.fit(self.X_train, self.y_train)
            return decision_tree.predict(X_test)
        
        elif self.regressor_name == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree = 2)  
            X_poly = poly_features.fit_transform(self.X_train)
            X_poly_test = poly_features.transform(X_test)
            poly_model = LinearRegression()  
            poly_model.fit(X_poly, self.y_train)
            return poly_model.predict(X_poly_test)
        
        elif self.regressor_name == 'Linear Regression':
            linear_regressor = LinearRegression()
            linear_regressor.fit(self.X_train, self.y_train)
            return linear_regressor.predict(X_test)
        
        elif self.regressor_name == "Random Forest":
            # Random trees uses many random decision trees to output many different results
            # we take the mean of the results
            random = RandomForestRegressor(n_estimators=100, max_depth = 2)
            random.fit(self.X_train, self.y_train)
            return random.predict(X_test)
        
        elif self.regressor_name == 'KNN Regression':
            neigh = KNeighborsRegressor(n_neighbors=4)
            neigh.fit(self.X_train, self.y_train)
            return neigh.predict(X_test)

            
  
    def getScore(self, y_test, y_pred):
        if self.regressor_name == 'Polynomial Regression':
            return -metrics.r2_score(y_test, y_pred)
        else: return metrics.r2_score(y_test, y_pred)
    
        
        
        