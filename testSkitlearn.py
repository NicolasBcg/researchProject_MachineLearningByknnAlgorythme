#classification for data with discrete label
#RadiusNeighborsClassifier
#KNeighborsClassifier
#a larger k suppresses the effects of noise, but makes the classification boundaries less distinct
#For high-dimensional parameter spaces,RadiusNeighborsClassifier becomes less effective
"""Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit
This can be accomplished through the weights keyword. 
The default value, weights = 'uniform', assigns uniform weights to each neighbor. 
weights = 'distance' assigns weights proportional to the inverse of the distance from the query point.
Alternatively, a user-defined function of the distance can be supplied to compute the weights"""

"""However, as the number of samples  grows, the brute-force approach quickly becomes infeasible.
 In the classes within sklearn.neighbors, brute-force neighbors searches are specified using the keyword algorithm = 'brute',
 and are computed using the routines available in sklearn.metrics.pairwise.

 K-dimensional tree  fast for low-dimensional (D<20) neighbors searches, it becomes inefficient as  grows very large: algorithm = 'kd_tree'

 ball tree makes tree construction more costly than that of the KD tree : algorithm = 'ball_tree'
 data structure which can be very efficient on highly structured data, even in very high dimensions.
"""

#For small data sets ( N less than 30 or so), brute force algorithms can be more efficient than a tree-based approach
"""Brute force query time is largely unaffected by the value of 
Ball tree and KD tree query time will become slower as  increases."""
#If very few query points will be required, brute force is better than a tree-based method


#tree based algorythm:leaf size parametre
"""A larger leaf_size leads to a faster tree construction time, because fewer nodes need to be created
 
  Both a large or small leaf_size can lead to suboptimal query cost. For leaf_size approaching 1,
  the overhead involved in traversing nodes can significantly slow query times. 
  For leaf_size approaching the size of the training set, queries become essentially brute force.
  A good compromise between these is leaf_size = 30, the default value of the parameter.

 As leaf_size increases, the memory required to store a tree structure decreases.
 This is especially important in the case of ball tree, which stores a -dimensional centroid for each node.
 The required storage space for BallTree is approximately 1 / leaf_size times the size of the training set.
"""

#possibles metrics for parametre DistanceMetric : print(sorted(KDTree.valid_metrics))
#use  NeighborhoodComponentsAnalysis to find the better space to apply knn to


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score,train_test_split


n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() -  0.3, X[:, 0].max() +  0.3
    y_min, y_max = X[:, 1].min() -  0.3, X[:, 1].max() +  0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    scores = cross_val_score(clf, X, y, cv=5)
    print("3-Class classification (k = %i, weights = '%s') %0.2f accuracy with a standard deviation of %0.2f" % (n_neighbors, weights,scores.mean(), scores.std()))



radius=1.5
for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.RadiusNeighborsClassifier(radius, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() +  0.3
    y_min, y_max = X[:, 1].min() -  0.3, X[:, 1].max() +  0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (r = %i, weights = '%s')" % (radius, weights)
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    scores = cross_val_score(clf, X, y, cv=5)
    print("3-Class classification (k = %i, weights = '%s') %0.2f accuracy with a standard deviation of %0.2f" % (n_neighbors, weights,scores.mean(), scores.std()))


plt.show()

