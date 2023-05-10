# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:58:01 2023

@author: tahat
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,r2_score


class PredictionModels:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load_data(self, filename):
        try:
            data = pd.read_csv(filename)
        except:
            try:
                data = pd.read_excel(filename) 
            except:
                try:
                    data = pd.read_table(filename) 
                except:
                    print("Dosya okunamadı!")
                    return None, None
        cols = list(data.columns)
        print("Veri sütunları: ")
        for i, col in enumerate(cols):
            print(f"{i+1}. {col}")
        independent_cols = input("Bağımsız değişkenlerin adlarını virgülle ayırarak girin: ").split(',')
        dependent_col = input("Bağımlı değişkenin adını girin: ")

        X = data[independent_cols].values
        y = data[dependent_col].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    def pred_model(self):
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == "lasso":
            alpha_value = float(input("Alpha değerini girin: "))
            self.model = Lasso(alpha=alpha_value)
        elif self.model_type == "ridge":
            alpha_value = float(input("Alpha değerini girin: "))
            self.model = Ridge(alpha=alpha_value)
        elif self.model_type == "elasticnet":
            alpha_value = float(input("Alpha değerini girin: "))
            l1_ratio_value = float(input("L1 ratio değerini girin: "))
            self.model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio_value)
        else:
            return None
    def train_model(self, X_train, y_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    def degerlendirme(self, y, y_pred):
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test,y_pred)
        print('R2 Skoru: ',r2)

class ClassificationModels:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load_data(self, filename):
        try:
            data = pd.read_csv(filename)
        except:
            try:
                data = pd.read_excel(filename) 
            except:
                try:
                    data = pd.read_table(filename) 
                except:
                    print("Dosya okunamadı!")
                    return None, None
        
        cols = list(data.columns)
        print("Veri sütunları: ")
        for i, col in enumerate(cols):
            print(f"{i+1}. {col}")
        independent_cols = input("Bağımsız değişkenlerin adlarını virgülle ayırarak girin: ").split(',')
        dependent_col = input("Bağımlı değişkenin adını girin: ")

        X = data[independent_cols].values
        y = data[dependent_col].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    def class_model(self):
        if self.model_type =='dtc':
            criterion = str(input("Lütfen kriter giriniz:(gini/entropy): "))
            splitter = str(input("Splitter giriniz(best/random): "))
            max_depth = int(input("Maksimum derinlik giriniz: "))
            self.model = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth)
        elif self.model_type == "rfc":
            n_estimators = int(input("Ağaç Sayısını giriniz(örn:3): "))
            criterion = str(input("Lütfen kriter giriniz:(gini/entropy): "))
            max_depth = int(input("Maksimum derinlik giriniz: "))
            self.model = RandomForestClassifier(n_estimators = n_estimators,criterion=criterion,max_depth=max_depth)
        elif self.model_type =="knn":
            n_neighbors = int(input("Komşu sayısını giriniz(örn:5): "))
            metric = str(input("Metrik türünü giriniz(örn: minkowkski): "))
            weights = str(input("Ağırlık türü giriniz(uniform/distance): "))
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,weights=weights)
        elif self.model_type == "gnb":
          prior_input = input("Öncül olasılıkları aralarında boşluk bırakarak giriniz (örn: 0.4 0.3): ")
          priors_list = list(map(float, prior_input.split()))
          priors = np.array(priors_list)
          var_smoothing = float(input("Var_smoothing değerini giriniz (örn: 1e-9): "))
          self.model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
        elif self.model_type == "svm":
            c_value = float(input("C değerini girin: "))
            kernel_value = str(input("Kernel türünü girin(örn: linear/polynomial/rbf/sigmoid): "))
            if kernel_value in ["polynomial", "linear"]:
                epsilon_value = float(input("Epsilon değerini girin: "))
                gamma_value = str(input("Gamma değerini girin: "))
                self.model = SVC(C=c_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value)
            elif kernel_value == "rbf":
                gamma_value = str(input("Gamma değerini girin: "))
                self.model = SVC(C=c_value, kernel=kernel_value, gamma=gamma_value)
            elif kernel_value == "sigmoid":
                gamma_value = str(input("Gamma değerini girin: "))
                coef_value = float(input("Coef0 değerini girin: "))
                self.model = SVC(C=c_value, kernel=kernel_value, gamma=gamma_value, coef0=coef_value)
            else:
                self.model = SVC(C=c_value, kernel=kernel_value)            
        else:
            return None
    def train_model2(self, X_train, y_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate_model2(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy*100))

class ClusteringModels:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load_data(self, filename):
        try:
            data = pd.read_csv(filename)
        except:
            try:
                data = pd.read_excel(filename) 
            except:
                try:
                    data = pd.read_table(filename) 
                except:
                    print("Dosya okunamadı!")
                    return None, None
        cols = list(data.columns)
        print("Veri sütunları: ")
        for i, col in enumerate(cols):
            print(f"{i+1}. {col}")
        independent_cols = input("Bağımsız değişkenlerin adlarını virgülle ayırarak girin: ").split(',')
        X = data[independent_cols].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def clustering_model(self):
        if self.model_type =="kmeans":
            cluster = int(input("Küme sayısını giriniz(örn:10): "))
            max_iter = int(input("İterasyon sayısını giriniz(örn:3):  "))
            self.model = KMeans(n_clusters=cluster,max_iter=max_iter, init ="k-means++")
        elif self.model_type =="agc":
            cluster = int(input("Küme sayısını giriniz(örn:10): "))
            self.model = AgglomerativeClustering(n_clusters=cluster,affinity="euclidean")
        elif self.model_type == "dbscan":
            distance = float(input("EPS değerini giriniz:(örn:0.5)"))
            sample = int(input("Minimum örnek sayısını giriniz(örn:2): "))
            metric = str(input("Metrik giriniz(örn:euclidean): "))
            self.model = DBSCAN(eps = distance, min_samples=sample,metric = metric)

    def train_model(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train)
    

    def evaluate_model(self, X_test):
        if self.model_type == "kmeans":
            y_pred = self.model.predict(X_test)
            score = silhouette_score(X_test, y_pred)
            print("Silhouette Score: {:.2f}".format(score))
        elif self.model_type == "agc":
            y_pred = self.model.fit_predict(X_test)
            score = silhouette_score(X_test, y_pred)
            print("Silhouette Score: {:.2f}".format(score))
        else:
            print("Silhouette Score hesaplanamıyor. Lütfen k-means veya agc modeli seçin.")

if __name__ == '__main__':

    secim = str(input("Lütfen algoritma Türü Seçiniz(Prediction, Classification , Clustering):"))
    if secim.lower() == "prediction":
            while True:
                model_prediction = ["lasso","linear","elasticnet","ridge"]
                print("Kullanmak istediğiniz modeli seçin: ")
                for i, model in enumerate(model_prediction):
                    print(f"{i+1}. {model}")
                model_type = input()
                if model_type.lower() not in model_prediction:
                    print("Yanlış seçim")
                else:
                    break
            model = PredictionModels(model_type)
            filename = input("Veri dosyasını giriniz: ")
            X,y = model.load_data(filename)
            test_size = float(input("Test verisi oranını girin (örn. 0.3): "))
            random_state = int(input("Random state değerini girin (örn. 42): "))
            
            X_train, X_test,y_train,y_test =  train_test_split(X,y, test_size=test_size, random_state=random_state)
            
            model.pred_model()
            model.train_model(X_train,y_train)
            model.degerlendirme(X_test,y_test)
            y_pred = model.predict(X_test)
    elif secim.lower() == "classification":
            while True:
                model_categorycal = ["dtc","rfc","knn","svm","gnb","svm"]
                print("Kullanmak istediğiniz modeli seçin: ")
                for i,model in enumerate(model_categorycal):
                    print(f"{i+1}. {model}")
                model_type = input()
                if model_type.lower() not in model_categorycal:
                    print("Yanlış seçim")
                else:
                    break
                
            model = ClassificationModels(model_type)
            filename = input("Veri dosyasını giriniz: ")
            X,y = model.load_data(filename)
            test_size = float(input("Test verisi oranını girin (örn. 0.3): "))
            random_state = int(input("Random state değerini girin (örn. 42): "))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model.class_model()
            model.train_model2(X_train,y_train)
            model.evaluate_model2(X_test,y_test)
            y_pred = model.predict(X_test)
    elif secim.lower() == "clustering":
        while True:
            models_clustering = ["dbscan","agc","kmeans"]
            print("Kullanmak istediğiniz modeli seçin: ")
            for i, model in enumerate(models_clustering):
                print(f"{i+1}. {model}")
            model_type = input()
            if model_type.lower() not in models_clustering:
                print("Yanlış seçim")
            else:
                break
        model = ClusteringModels(model_type)
        filename = input("Veri dosyasını giriniz: ")
        X = model.load_data(filename)
        test_size = float(input("Test verisi oranını girin (örn. 0.3): "))
        random_state = int(input("Random state değerini girin (örn. 42): "))
        X_train, X_test =  train_test_split(X, test_size=test_size, random_state=random_state)
        model.clustering_model()
        model.train_model(X_train)
        model.evaluate_model(X_train) 
    else:
            print("Girdi kabul edilmedi")