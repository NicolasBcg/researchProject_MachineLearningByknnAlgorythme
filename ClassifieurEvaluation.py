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

# create new file for the output
fichierSortie = open("data.txt", "a")
fichiersDonnees = os.listdir("./datas")

nbfichiers=20
nbPlis=5
nbFichiersParPlis= int(nbfichiers/nbPlis)

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
for pca_n_features in [12]:
    pca = PCA(n_components=pca_n_features)
    for n_neighbors in [20]:
        print("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
        fichierSortie.write("\n --------------------------- \n Class classification (k = %i, weights = '%s', pca features= %d , nb jeux de donnees = %d ) " % (n_neighbors, weights,pca_n_features,nbfichiers))
        totalTrainingAccuracy=0
        totalTestingAccuracy=0
        précisionParOrganes=[0.0]*23 #23 est le nombre de caractéristiques
        recallParOrganes=[0.0]*23 #23 est le nombre de caractéristiques
        start = time.time()  
        for i in range(nbPlis):
            print("\n --------------------------- \n Pli n° %d ) " % (i+1))
            fichierSortie.write("\n --------------------------- \n Pli n° %d ) " % (i+1))
            print("\n Extraction des donnees en cours ")
            startExtract = time.time()
            if (i==0):
                print("train :")
                with open('./datas/'+fichiersDonnees[nbfichiers-1]) as file:
                    print(nbfichiers-1)
                    X_train=np.loadtxt(file, delimiter=",")

                for fichier in range(nbFichiersParPlis*(i+1),nbfichiers-1):
                    print(fichier)
                    with open('./datas/'+fichiersDonnees[(i*nbFichiersParPlis)+fichier]) as file:
                        listesDonnees=np.loadtxt(file, delimiter=",")
                        X_train=np.concatenate((listesDonnees,X_train))  

            if (i!=0):
                print("train :")
                with open('./datas/'+fichiersDonnees[0]) as file:
                    print(0)
                    X_train=np.loadtxt(file, delimiter=",")  

                for fichier in range(1,nbFichiersParPlis*i):
                    print(fichier)
                    with open('./datas/'+fichiersDonnees[fichier]) as file:
                        listesDonnees=np.loadtxt(file, delimiter=",")
                        X_train=np.concatenate((listesDonnees,X_train))


                for fichier in range(nbFichiersParPlis*(i+1),nbfichiers):
                    
                    print(fichier)
                    with open('./datas/'+fichiersDonnees[fichier]) as file:
                        listesDonnees=np.loadtxt(file, delimiter=",")
                        X_train=np.concatenate((listesDonnees,X_train))  
            endExtract=time.time()
            start+=(endExtract-startExtract)
            print("\n Fin de l'extraction des donnees duree %d", (endExtract-startExtract))
            fichierSortie.write("\n Duree de l'extraction des donnees : %d" % (endExtract-startExtract))


            y_train=np.ravel(X_train[:,:1])
            X_train=X_train[:,1:]

            
            #start = time.time()       
            pca.fit(X_train, y_train)
            print("pca done")
            """end = time.time()
            print("PCA time = %d" % (end-start))
            fichierSortie.write("\nPCA time = %d" % (end-start))"""
            X_train=pca.transform(X_train)
            print("pca on X_train done")
            with open('./datas/'+fichiersDonnees[i*nbFichiersParPlis]) as file:
                print(i*nbFichiersParPlis)    
                X_test=np.loadtxt(file, delimiter=",")   
            for fichier in range((nbFichiersParPlis*i)+1,nbFichiersParPlis*(i+1)):
                print("test :")
                print(fichier)
                with open('./datas/'+fichiersDonnees[fichier]) as file:
                    listesDonnees=np.loadtxt(file, delimiter=",")
                    X_test=np.concatenate((listesDonnees,X_test))      

            y_test=np.ravel(X_test[:,:1])
            X_test=X_test[:,1:]

            knn = KNeighborsClassifier(n_neighbors, weights=weights)
            knn.fit(X_train, y_train)
            
            # test the training poin
            
            #start = time.time()
            y_pred=knn.predict(X_train)       
            #end = time.time()
            testingScores=accuracy_score(y_pred,y_train)            
            """print("On training datas %0.3f accuracy  ||| testing time = %d" % (testingScores.mean(),(end-start)))
            fichierSortie.write("On training datas %0.3f accuracy  ||| testing time = %d" % (testingScores.mean(),(end-start)))
            """
            totalTrainingAccuracy+=testingScores.mean()
            
            # test the testing points 
            #start = time.time()
            y_pred=knn.predict(pca.transform(X_test))       
            #end = time.time()
            testingScores=accuracy_score(y_pred,y_test)
            """print("On testing datas %0.3f accuracy ||| testing time = %d" % (testingScores.mean(),(end-start)))
            fichierSortie.write("\nOn testing datas %0.3f accuracy ||| testing time = %d" % (testingScores.mean(),(end-start)))"""
            totalTestingAccuracy+=testingScores.mean()


            matConf=confusion_matrix(y_test, y_pred)
            y_test=[]
            X_test=[]
            (l,a)=matConf.shape
            for i in range (l):
                totalprécision=0
                totalRecall=0
                for j in range (l):
                    totalprécision+=matConf[i][j]
                    totalRecall+=matConf[j][i]
                précisionParOrganes[i]+=matConf[i][i]/totalprécision
                recallParOrganes[i]+=matConf[i][i]/totalRecall
            matConf=[]
        end = time.time()
            #fichierSortie.write("\nmatrice de confusion\n")
            #fichierSortie.write(np.array_str(matConf))
        for i in range(len(précisionParOrganes)):
            précisionParOrganes[i]=round(précisionParOrganes[i]/nbPlis,3)
            recallParOrganes[i]=round(recallParOrganes[i]/nbPlis,3)
        
        print("\nPrécision :\n")
        print(précisionParOrganes)
        fichierSortie.write("\nPrécision :\n")
        fichierSortie.write(''.join(map(convert, précisionParOrganes)))
        print("\n Recall :\n")
        print(recallParOrganes)
        fichierSortie.write("\n Recall :\n")
        fichierSortie.write(''.join(map(convert, recallParOrganes)))


        print("\n --------------------------- \n On training datas %0.3f accuracy ||| On testing datas %0.3f accuracy ||| testing time = %d " % (totalTrainingAccuracy/nbPlis,totalTestingAccuracy/nbPlis,(end-start)))      
        fichierSortie.write("\n --------------------------- \n On training datas %0.3f accuracy ||| On testing datas %0.3f accuracy ||| testing time = %d" % (totalTrainingAccuracy/nbPlis,totalTestingAccuracy/nbPlis,(end-start)))
fichierSortie.close()
