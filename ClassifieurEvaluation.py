import numpy as np
import time
import os
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix

def convert(i):
    return str(i)+' ; '


# import some data to play with
#datas = datasets.load_iris()
#digits
fichierSortie = open("data.txt", "a")
#datas=datasets.load_digits()
fichiersDonnees = os.listdir("./datas")
listesDonnees=[]
print("extraction des données")

nbfichiers=20
for fichier in range(nbfichiers):
    with open('./datas/'+fichiersDonnees[fichier]) as file:
        listesDonnees.append(np.loadtxt(file, delimiter=","))
    print(str(fichier+1)+" fichiers extraits...")
nbPlis=5

AllTrainingDatas = np.concatenate(listesDonnees[:nbfichiers])

trainingDatas=[]#stock the training datas for each fold
testingDatas=[]#stock the testing datas for each fold
nbFichiersParPlis= int(nbfichiers/nbPlis)
for i in range(nbPlis):
    train=np.concatenate(listesDonnees[0:nbFichiersParPlis*i]+listesDonnees[nbFichiersParPlis*(i+1):nbfichiers])
    trainingDatas.append([train[:,1:],np.ravel(train[:,:1])])       #[x_train,y_train]
    test=np.concatenate(listesDonnees[nbFichiersParPlis*i:nbFichiersParPlis*(i+1)])
    testingDatas.append([test[:,1:],np.ravel(test[:,:1])])      #[x_test,y_test]




# Without PCA (Classic knn)
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
fichierSortie.write("\n --------------------------- \n Class classification with PCA")


"""
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

weights = "distance"
for pca_n_features in [18]:
    pca = PCA(n_components=pca_n_features)
    for n_neighbors in [10,12,14,18]:
        # we create an instance of Neighbours Classifier and fit the data.
        print("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
        fichierSortie.write("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
        totalTrainingAccuracy=0
        totalTestingAccuracy=0
        for i in range(nbPlis):
            print("\n --------------------------- \n Pli n° %d ) " % (i+1))
            fichierSortie.write("\n --------------------------- \n Pli n° %d ) " % (i+1))
            X_train=trainingDatas[i][0]
            y_train=trainingDatas[i][1]
            X_test=testingDatas[i][0]
            y_test=testingDatas[i][1]
            start = time.time()
            
            pca.fit(X_train, y_train)
            end = time.time()
            print("PCA time = %d" % (end-start))
            fichierSortie.write("\nPCA time = %d" % (end-start))
            knn = KNeighborsClassifier(n_neighbors, weights=weights)
            knn.fit(pca.transform(X_train), y_train)
            
            # test the training point
            start = time.time()
            y_pred=knn.predict(pca.transform(X_train))       
            end = time.time()
            testingScores=accuracy_score(y_pred,y_train)            
            print("On training datas %0.3f accuracy  ||| testing time = %d" % (testingScores.mean(),(end-start)))
            fichierSortie.write("On training datas %0.3f accuracy  ||| testing time = %d" % (testingScores.mean(),(end-start)))

            totalTrainingAccuracy+=testingScores.mean()
            
            # test the testing points 
            start = time.time()
            y_pred=knn.predict(pca.transform(X_test))       
            end = time.time()
            testingScores=accuracy_score(y_pred,y_test)
            print("On testing datas %0.3f accuracy ||| testing time = %d" % (testingScores.mean(),(end-start)))
            fichierSortie.write("\nOn testing datas %0.3f accuracy ||| testing time = %d" % (testingScores.mean(),(end-start)))
            totalTestingAccuracy+=testingScores.mean()


            matConf=confusion_matrix(y_test, y_pred)
            matConfproportion=[]
            (l,a)=matConf.shape
            for i in range (l):
                total=0
                for j in range (l):
                    total+=matConf[i][j]
                matConfproportion.append(matConf[i][i]/total)
            print("matrice de confusion")
            print(matConf)
            print("proportion de vrais justes:")
            print(matConfproportion)
            #fichierSortie.write("\nmatrice de confusion\n")
            #fichierSortie.write(np.array_str(matConf))
            fichierSortie.write("\nen proportion :\n")
            fichierSortie.write(''.join(map(convert, matConfproportion)))

        print("\n --------------------------- \n On training datas %0.3f accuracy ||| On testing datas %0.3f accuracy " % (totalTrainingAccuracy/nbPlis,totalTestingAccuracy/nbPlis))      
        fichierSortie.write("\n --------------------------- \n On training datas %0.3f accuracy ||| On testing datas %0.3f accuracy " % (totalTrainingAccuracy/nbPlis,totalTestingAccuracy/nbPlis))
fichierSortie.close()
