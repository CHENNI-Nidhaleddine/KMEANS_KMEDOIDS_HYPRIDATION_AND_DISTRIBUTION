# server.py
import Pyro4
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import normalized_mutual_info_score
import time
import numpy as np


class MessageReceiver(object):
    def __init__(self):
        self.clients = {}
        self.clients_new = {}
        self.clients_continue={}
        self.convergenve=False
        self.data=[]
        self.labels=[]
        self.trues=[]
        self.n_clusters=0
        self.featuresSize=0
        self.medoids=[]
        self.inertia=float('inf')
        self.labelsss=[]
        self.resultFromClient=0
        self.start=0
        self.end=0

    def setData(self,data):
        self.data=data
    def setLabels(self,labels):
        self.labels=labels
    def setK(self,k):
        self.n_clusters=k


    @Pyro4.expose
    def getData(self):
        return self.data.tolist()
    
    @Pyro4.expose
    def getLabels(self):
        return self.labels
    
    @Pyro4.expose
    def getK(self):
        return self.n_clusters
    
    @Pyro4.expose
    def getMedoids(self):
        return self.medoids
    
    @Pyro4.expose
    def getClientNew(self,id):
        return self.clients_new[id]
    
    @Pyro4.expose
    def setClientNew(self,id):
        self.clients_new[id]=False
    
    def setLabels(self,labels):
        self.labels=labels
    def setTrues(self,labels):
        self.trues=labels
    
    def setFeaturesSize(self,s):
        self.featuresSize=s

    @Pyro4.expose
    def getLabels(self):
        return self.labels
    
    @Pyro4.expose
    def CanContinue(self,client_id):
        return self.clients_continue[client_id]
    

    @Pyro4.expose
    def register_client(self, client_id):
        self.clients[client_id] = None
        self.clients_new[client_id]=True
        self.clients_continue[client_id]=True
    @Pyro4.expose
    def getFeaturesSize(self):
        return self.featuresSize

    @Pyro4.expose
    def sendResult(self,labels,inertia,id):

        if(inertia<self.inertia):
            self.labels=np.array(labels)
            medoids=get_medoid_indices(self.data, self.labels)
            kmedoids_instance = kmedoids(self.data, medoids,k=self.n_clusters,max_iter=8)
            kmedoids_instance.process()

            # Get the final medoids and cluster assignments
            final_medoids = kmedoids_instance.get_medoids()
            self.labelsss = kmedoids_instance.get_clusters()
            self.inertia=inertia
            print(inertia)
            
            # no updates between two clients
            if(final_medoids==self.medoids):
                self.resultFromClient=self.resultFromClient+1
                labelss=[0]*569
                for j in range(2):
                    for i in range(len(self.labelsss[j])):
                        labelss[self.labelsss[j][i]]=j
                nmi_score = normalized_mutual_info_score(self.labels, labelss)
                print(nmi_score)
                if(self.resultFromClient==70):
                    self.convergenve=True


            self.medoids=final_medoids
        
    @Pyro4.expose
    def getConvergence(self):
        return self.convergenve



#utils

def get_medoids(X, labels, centroids):
    medoids = []
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        distances = pairwise_distances_argmin_min(cluster_points, cluster_points)
        medoid_index = np.argmin(np.sum(distances[1]))
        medoid = cluster_points[medoid_index]
        medoids.append(medoid)
        
    return medoids
def get_medoid_indices(X, labels):
    labels = labels.astype(int)  # Convert labels to integers
    medoid_indices = []
    for k in range(np.max(labels) + 1):
        cluster_points = X[labels == k]
        distances = pairwise_distances_argmin_min(cluster_points, cluster_points)
        medoid_index = np.argmin(np.sum(distances[1]))
        medoid_indices.append(np.where(labels == k)[0][medoid_index])
    return medoid_indices



# Just conncetions

# Create a Pyro daemon
daemon = Pyro4.Daemon()


# Register the MessageReceiver object with Pyro
receiver = MessageReceiver()

#Setting the data


#B=0
#M=1
# data = pd.read_csv("Iris.csv")
# receiver.setData(data.values[:,:-1])
# receiver.setLabels(data.Species.map({"Iris-setosa" : 1, "Iris-virginica": 0, "Iris-versicolor": 2}))
# receiver.setK(3)

data = pd.read_csv("data.csv")

receiver.setData(data.values[:,2:])
# print(receiver.data)
# print("--------------")
receiver.setTrues(data.values[:,1])
receiver.setLabels(data.values[:,1])
# print(receiver.labels)
receiver.setK(2)
receiver.setFeaturesSize(30)
uri = daemon.register(receiver)


# Print the URI for the clients to connect
print("Server URI:", uri)
def calculate_inertia(labels, data):
    kmeans = KMeans(n_clusters=len(set(labels)))
    kmeans.fit(data)
    inertia = kmeans.inertia_
    return inertia

# Start the Pyro event loop
daemon.requestLoop()
