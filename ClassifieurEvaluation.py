import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA



# import some data to play with
#datas = datasets.load_iris()
#digits

#datas=datasets.load_digits()
fichiersDonnees = os.listdir("./datas")
listesDonnees=[]
print("extraction des données")

nbfichiers=5

for fichier in range(nbfichiers):
    with open('./datas/'+fichiersDonnees[fichier]) as file:
        listesDonnees.append(np.loadtxt(file, delimiter=","))
    print(str(fichier+1)+" fichiers extraits...")


datas = np.concatenate(listesDonnees)




X=datas[:,1:]
y=np.ravel(datas[:,:1])
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Without PCA (Classic knn) takes too much time with numerous datas
"""print("\n --------------------------- \n Class classification without NCA")
for weights in ["uniform", "distance"]:
    for n_neighbors in [10,15]:
        print("\n --------------------------- \n Class classification (k = %i, weights = '%s') " % (n_neighbors, weights))
        # we create an instance of Neighbours Classifier and fit the data.
        start = time.time()
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_train, y_train)
        end = time.time()
        print("training time = %d" % (end-start))
        # test the training points and the testing points  
        start = time.time()
        CrossValidationScores = cross_val_score(clf,X_train, y_train, cv=5)
        end = time.time()
        print("On training datas %0.3f accuracy with a standard deviation of %0.2f " % (CrossValidationScores.mean(), CrossValidationScores.std()))
        print(" evaluation time = %d" % (end-start))
        start = time.time()
        testingScores=clf.score(X_test, y_test)
        end = time.time()
        print("On testing datas %0.3f accuracy with a standard deviation of %0.2f " % (testingScores.mean(), testingScores.std()))
        print("testing time = %d" % (end-start))
"""
# With PCA
print("\n --------------------------- \n Class classification with PCA")

for weights in ["distance","uniform"]:
    for pca_n_features in [5,8,10]:
        for n_neighbors in [10,13,15]:

            # we create an instance of Neighbours Classifier and fit the data.
            print("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
            print("processing pca...")
            start = time.time()
            pca = PCA(n_components=pca_n_features)
            pca.fit(X_train, y_train)
            end = time.time()
            print("pcadone time = %d" % (end-start))     
            knn = KNeighborsClassifier(n_neighbors, weights=weights)
            start = time.time()
            knn.fit(pca.transform(X_train), y_train)
            end = time.time()
            print("training time = %d" % (end-start))

            # test the training points and the testing points 
            start = time.time()
            CrossValidationScores = cross_val_score(knn,pca.transform(X_train), y_train, cv=5)
            end = time.time()
            print(" On training datas %0.3f accuracy with a standard deviation of %0.2f ||| evaluation time = %d" % (CrossValidationScores.mean(), CrossValidationScores.std(),(end-start)))

            start = time.time()
            testingScores=knn.score(pca.transform(X_test), y_test)
            end = time.time()
            print("On testing datas %0.3f accuracy with a standard deviation of %0.2f  ||| testing time = %d" % (testingScores.mean(), testingScores.std(),(end-start)))

"""

weights = "uniform"
for pca_n_features in [8]:
    for n_neighbors in range(1,11):

        # we create an instance of Neighbours Classifier and fit the data.
        print("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
        print("processing pca...")
        start = time.time()
        pca = PCA(n_components=pca_n_features)
        pca.fit(X_train, y_train)
        end = time.time()
        print("pcadone time = %d" % (end-start))     
        knn = KNeighborsClassifier(n_neighbors, weights=weights)
        start = time.time()
        knn.fit(pca.transform(X_train), y_train)
        end = time.time()
        print("training time = %d" % (end-start))

        # test the training points and the testing points 
        start = time.time()
        CrossValidationScores = cross_val_score(knn,pca.transform(X_train), y_train, cv=5)
        end = time.time()
        print(" On training datas %0.3f accuracy with a standard deviation of %0.2f ||| evaluation time = %d" % (CrossValidationScores.mean(), CrossValidationScores.std(),(end-start)))

        start = time.time()
        testingScores=knn.score(pca.transform(X_test), y_test)
        end = time.time()
        print("On testing datas %0.3f accuracy with a standard deviation of %0.2f  ||| testing time = %d" % (testingScores.mean(), testingScores.std(),(end-start)))

       """ 