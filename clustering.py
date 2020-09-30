from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, algorithm_cluster, X_train):
        self.algorithm_cluster = algorithm_cluster
        self.X_train = X_train
        
    def predict(self, X_test):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.X_train)
        return kmeans.predict(X_test)
