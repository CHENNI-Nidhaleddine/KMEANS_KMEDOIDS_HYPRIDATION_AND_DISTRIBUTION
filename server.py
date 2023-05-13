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

import numpy as np


class MessageReceiver(object):
    def __init__(self):
        self.clients = {}
        self.clients_new = {}
        self.clients_continue={}
        self.convergenve=False
        self.data=[]
        self.labels=[]
        self.n_clusters=0
        self.medoids=[]
        self.inertia=9999999
        self.labelsss=[]

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
    def sendResult(self,labels,inertia,id):
        # self.clients_continue[id]=False
        if(inertia<self.inertia):
            self.labels=np.array(labels)
            medoids=get_medoid_indices(self.data, self.labels)
            kmedoids_instance = kmedoids(self.data, medoids,k=3,max_iter=1)
            kmedoids_instance.process()
            # self.clients_new[id]=False

            # Get the final medoids and cluster assignments
            final_medoids = kmedoids_instance.get_medoids()
            self.labelsss = kmedoids_instance.get_clusters()
            self.inertia=inertia
            print(inertia)
            
            # no updates between two clients
            # if(final_medoids==self.medoids):
            #     self.convergenve=True
            #     labelss=[0]*150
            #     for j in range(3):
            #         for i in range(len(self.labelsss[j])):
            #             labelss[self.labelsss[j][i]]=j
            #     nmi_score = normalized_mutual_info_score(self.labels, labelss)
            #     print(nmi_score)

            self.medoids=final_medoids
            

        # no update in inertia
        elif(inertia==self.inertia):
            labelss=[0]*150
            self.convergenve=True # to stop clients
            for j in range(3):
                for i in range(len(self.labelsss[j])):
                    labelss[self.labelsss[j][i]]=j
            nmi_score = normalized_mutual_info_score(self.labels, labelss)
            print(nmi_score)
        
        # self.clients_continue[id]=True

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
data = pd.read_csv("Iris.csv")
receiver.setData(data.values[:,:-1])
receiver.setLabels(data.Species.map({"Iris-setosa" : 1, "Iris-virginica": 0, "Iris-versicolor": 2}))
receiver.setK(3)

uri = daemon.register(receiver)


# Print the URI for the clients to connect
print("Server URI:", uri)

# Start the Pyro event loop
daemon.requestLoop()
