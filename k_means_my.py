# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:41:03 2017

@author: alexey
"""

import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans

class MyKMeans:
    def __init__(self, n_clusters=2, metric='euclidean', max_iter=300):
        '''
        n_clusters - число кластеров
        metric - метрика
        max_iter - максимальное число итераций
        '''
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.centers = []


    @staticmethod
    def distance(vector1, vector2):
        '''
        Определяем функцию расстояния
        '''
        dist_d = 0
        
        date_d = list(map(lambda pair:(pair[0]-pair[1])**2,zip(vector1,vector2)))
        
        for x in date_d: 
            dist_d += x
        dist_d = np.sqrt(dist_d)
        return dist_d
        
    
    def predict(self, X):
        '''
        Предсказываем попадание объектов из X в конкретный кластер
        '''
        list_data = []
        print(" centers ", len(self.centers), " x ", len(X))
        print(self.centers)
        for x in X:                
            dist_data = [self.distance(x,c) for c in self.centers]
            list_data.append(dist_data.index(min(dist_data)))
                    
        return list_data
        

    def fit(self, X):  
        
        '''
        Шаг 1 - Инизиализируем начальные положения центров кластеров
        '''
        try:
            centers_index = np.random.choice([x for x in range(len(X))],self.n_clusters, replace = False)
        
        except ValueError:
            print("number of cluster not acceptable")
            raise ValueError
        
        centers = [z for z_index,z in enumerate(X) for i in centers_index if z_index == i ]

        print("random center ", centers)           
        '''
        Шаг 2 - Выполняем уточнение положения центров кластеров до тех пор, пока 
        не будет превышено значение max_iter или центры кластеров не будут меняться 
        '''
        for step in range(self.max_iter):            
            '''
            Шаг 2.1 - Вычисляем расстояние до центров кластеров
            '''
            list_dist = []
            for x in X:
                list_dist.append([self.distance(c,x) for c in centers])
                    
            #print(list_dist)
            '''
            Шаг 2.2 - Для каждого объекта находим argmin от расстояний до центров
            '''
            y_i = [centers[x.index(min(x))] for x in list_dist]
            y_i = list(y_i)          
         
            '''
            Шаг 2.3 - Уточняеням положения центров кластеров
            '''
            new_centers = []
            
            for index_c,c in enumerate(centers):
                min_dist = []
                for index_y,y in enumerate(y_i):
                    if list(y) == list(c): 
                        min_dist.append(X[index_y])
                new_centers.append(list(map(lambda x: sum(x)/len(x), zip(*min_dist))))
                min_dist.clear()
            
            centers = np.array(new_centers)

            
        '''
        Шаг 3 - Сохраняем положения центров кластеров
        ''' 
        self.centers = centers
        
        '''
        Шаг 4 - Возвращяем предсказание
        '''        
        return self.predict(X)
        
        
my_data = np.genfromtxt('dataset3.csv')       
        
kmeans = KMeans(init='random', n_clusters=3, random_state=0)
        
proc_data = kmeans.fit_predict(my_data)       

mY_k =  MyKMeans(n_clusters=3)

my_kmeans = mY_k.fit(my_data)
        
my_proc_data = my_kmeans

print(" ski my  ", my_proc_data)

print(" ski ", proc_data, " k means ", kmeans.cluster_centers_)


import matplotlib.pyplot as plt

import matplotlib.image as mpimg

image = mpimg.imread('./mailru.jpg')

plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()

data = image.reshape((image.shape[0]*image.shape[1],3))

kmeans = MyKMeans(n_clusters=64)


my_kmeans = kmeans.fit(data)

new_image = []
for cluster in my_kmeans:
    new_image.append(kmeans.cluste[cluster])
    
    
new_image = new_image.reshape((image.shape[0],image.shape[1],3))


plt.axis("off")
plt.imshow(new_image)
plt.show()

#print(" means ", my_proc_data, len(my_proc_data))


