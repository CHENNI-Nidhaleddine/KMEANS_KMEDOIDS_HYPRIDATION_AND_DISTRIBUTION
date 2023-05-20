# client.py
import Pyro4
import uuid
import random
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extra.cluster import KMedoids

from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np;




# Retrieve the server URI from the server machine
# server_uri = input("Enter the server URI: ")
server_uri= "PYRO:obj_69f5a5249ee34edbae2703ee713b5390@localhost:53058"

# Generate a unique ID for this client
client_id = str(uuid.uuid4())

# Create a Pyro proxy to the server object
receiver = Pyro4.Proxy(server_uri)

# Register the client with the server
receiver.register_client(client_id)
X=np.array(receiver.getData())
k=receiver.getK()
labels=np.ones(X.shape[0])
centers=np.ones((k,receiver.getFeaturesSize()))
inertia=np.ones(1)
centers=X[np.random.choice(range(len(X)), size=k, replace=False)]

#get data
while(not receiver.getConvergence()):
    kmeans = KMeans(n_clusters=k,init=np.array(centers),max_iter=1)
    kmeans.fit(X)
    labels=kmeans.labels_
    centers= kmeans.cluster_centers_
    inertia=kmeans.inertia_
    centers=kmeans.cluster_centers_.tolist()
    labelss=kmeans.labels_.tolist()
    inertia=kmeans.inertia_
    receiver.sendResult(labelss,inertia,client_id)