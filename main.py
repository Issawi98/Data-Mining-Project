import pandas as pd
import numpy as np
from preprocessing import Preprocessor
from classification import Classifier
from regression import Regression
from clustering import Clustering
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *


def buttonClick():
    classificationAlg = variable1.get()
    regressionAlg = variable2.get()
    clusterAlg = variable3.get()
    
    classificationScaler = box1s.get()
    regressionScaler = box2s.get()
    clusterScaler = box3s.get()
    
    classificationMethod = box1F.get()
    regressionMethod = box2F.get()

    
    preprocessing = Preprocessor()
    
    # Classification
    classificationData = pd.read_csv('CSV/cancer.csv')
    X_class, y_class = preprocessing.dataCleaning(classificationData, 'cancer')
    X_class, y_class = preprocessing.encode(X_class, y_class, 'cancer')
    if classificationMethod == "drop":
        classificationData = preprocessing.drop(classificationData)
    elif classificationMethod == "replaceMean":
        classificationData = preprocessing.replaceMean(classificationData)
    elif classificationMethod == "replaceMode":
        classificationData = preprocessing.replaceMode(classificationData)
    else:
        classificationData = preprocessing.drop(classificationData)
    X_train_class, X_test_class, y_train_class, y_test_class = preprocessing.split(X_class, y_class, 0.3)
    X_train_class, X_test_class = preprocessing.scale(X_train_class, X_test_class, type = classificationScaler)
    classifier = Classifier(classificationAlg, X_train_class,np.ravel( y_train_class))
    y_pred_class = classifier.classify(X_test_class)
    accuracy = classifier.calculateAccuracy(np.ravel(y_test_class), y_pred_class)
    
    # Regression
    regressionData = pd.read_csv('CSV/diamonds.csv')
    X_regression, y_regression = preprocessing.dataCleaning(regressionData,'diamonds' )
    X_regression, y_regression = preprocessing.encode(X_regression, y_regression, 'diamonds')
    if regressionMethod == "drop":
        regressionData = preprocessing.drop(regressionData)
    elif regressionMethod == "replaceMean":
        regressionData = preprocessing.replaceMean(regressionData)
    elif regressionMethod == "replaceMode":
        regressionData = preprocessing.replaceMode(regressionData)
    else:
        regressionData = preprocessing.drop(classificationData)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessing.split(X_regression, y_regression, 0.3)
    X_train_reg, X_test_reg = preprocessing.scale(X_train_reg, X_test_reg, type = regressionScaler)
    regression = Regression(regressionAlg, X_train_reg, np.ravel(y_train_reg))
    y_pred_regression = regression.predict(X_test_reg)
    score = regression.getScore(np.ravel(y_test_reg), y_pred_regression)
    
    # Clustering
    X_clust, y_clust = load_iris(return_X_y = True)
    iris = sns.load_dataset("iris")
    X_train_clust, X_test_clust = preprocessing.splitCluster(X_clust, 0.3)   
    X_train_clust, X_test_clust = preprocessing.scale(X_train_clust, X_test_clust, type = clusterScaler)
    cluster = Clustering(clusterAlg, X_train_clust)
    y_pred_cluster = cluster.predict(X_test_clust)
    
    
    print('\n\nThe accuracy is: ' + str(accuracy) + ' %  ' + str(classificationAlg))
    print('\n\nThe score: ' + str(score) + ' ' + str(regressionAlg))
    print('\n\nPredicted clusters: ' + str(y_pred_cluster) + ' ' + str(clusterAlg))
    
    
    content = "The accuracy is:" + str(accuracy) + "%   using " + str(classificationAlg)+" Algorithm \n\n"+"The score is: " + str(score) + " using " + str(regressionAlg)+" Algorithm  \n\n"+ "Predicted clusters are " + str(y_pred_cluster) + " using " + str(clusterAlg) + " Algorithm  \n\n"
    text1 = Text(window, width =80, height=8)
    text1.grid(row=9, column=0, columnspan=30)
    text1.insert(INSERT, content)
    sb1 = Scrollbar(window)
    sb1.grid(row=9, column=5,  columnspan=40)
    text1.configure(yscrollcommand=sb1.set)
    sb1.configure(command=text1.yview)
    window.update()
    # Construct iris plot
    sns.swarmplot(x="species", y="petal_length", data=iris)
    # Show plot
    plt.show()
    
    sns.set(style="whitegrid")
    sns.boxplot(data=iris, orient="h", palette="Set2")
    plt.show()
    
    sns.set(style="whitegrid")
    sns.pairplot(iris)
    plt.show()
    

window = Tk()
window.geometry("877x320")
window.title("DM")
window.resizable(False, False)

variable1 = StringVar(window)
variable1.set("KNN")  # default value
l1 = Label(window, text="Classifier Method: ", width=20)
l1.grid(row=0, column=0)
box1 = OptionMenu(window, variable1, "KNN", "Decision Tree", "Naive Bayes", "Random Forest")
box1.grid(row=0, column=1)
l1s = Label(window, text="Scaling Method: ", width=20)
l1s.grid(row=0, column=2)
box1s = Entry(window)
box1s.grid(row=0, column=3)
box1s.insert(END, "StandardScaler")

l1F = Label(window, text="Missing Values: ", width=20)
l1F.grid(row=0, column=4)
box1F = Entry(window)
box1F.insert(END, "drop")
box1F.grid(row=0, column=5)



variable2 = StringVar(window)
variable2.set("Linear Regression")  # default value
l2 = Label(window, text="Regression Method: ", width=20)
l2.grid(row=1, column=0)
box2 = OptionMenu(window, variable2, "Linear Regression", "Decision Tree", "Polynomial Regression", "KNN Regression", "Random Forest")
box2.grid(row=1, column=1)
l2s = Label(window, text="Scaling Method: ", width=20)
l2s.grid(row=1, column=2)
box2s = Entry(window)
box2s.insert(END, "StandardScaler")
box2s.grid(row=1, column=3)

l2F = Label(window, text="Missing Values: ", width=20)
l2F.grid(row=1, column=4)
box2F = Entry(window)
box2F.insert(END, "drop")
box2F.grid(row=1, column=5)




variable3 = StringVar(window)
variable3.set("K-Means")  # default value

l3 = Label(window, text="Clustering Method: ", width=20)
l3.grid(row=2, column=0)
box3 = OptionMenu(window, variable3, "K-Means")
box3.grid(row=2, column=1)
l3s = Label(window, text="Scaling Method: ", width=20)
l3s.grid(row=2, column=2)
box3s = Entry(window)
box3s.insert(END, "StandardScaler")
box3s.grid(row=2, column=3)


l3F = Label(window, text="Missing Values: ", width=20)
l3F.grid(row=2, column=4)
box3F = Entry(window, state='disabled')
box3F.insert(END, "drop")
box3F.grid(row=2, column=5)




l4 = Label(window, text="")
l4.grid(row=3, column=1, columnspan=2)

b1 = Button(window, text="Submit", width=40,fg="white" ,bg="blue",command=buttonClick)
b1.grid(row=5, column=2, columnspan=2)

l6 = Label(window, text="")
l6.grid(row=6, column=1, columnspan=2)

l5 = Label(window, text="***Scaling Methods are: StandardScaler, MinMaxScaler,or MaxScaler")
l5.grid(row=7, column=0, columnspan=8)
l5 = Label(window, text="***We deal with missing values by: drop, replaceMean,or replaceMode")
l5.grid(row=8, column=0, columnspan=8)


text1 = Text(window, width =80, height=7)
text1.grid(row=9, column=0, columnspan=30)


window.mainloop()





