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

labels=np.ones(150)
centers=np.ones((3,5))
inertia=np.ones(1)


# Retrieve the server URI from the server machine
# server_uri = input("Enter the server URI: ")
server_uri= "PYRO:obj_95d6936243834cceb4f020dfc85013ec@localhost:52149"

# Generate a unique ID for this client
client_id = str(uuid.uuid4())

# Create a Pyro proxy to the server object
receiver = Pyro4.Proxy(server_uri)

# Register the client with the server
receiver.register_client(client_id)
X=np.array(receiver.getData())
k=receiver.getK()
centers=X[np.random.choice(range(len(X)), size=3, replace=False)]
new=receiver.getClientNew(client_id)

#get data
while(not receiver.getConvergence()):
    # print(centers)
    kmeans = KMeans(n_clusters=k,init=np.array(centers),max_iter=1)

    # if(new):
        # kmeans = KMeans(n_clusters=k,init=X[np.random.choice(range(len(X)), size=k, replace=False)],max_iter=1)
    # else:
    #     final_medoids=np.array(receiver.getMedoids())
        # kmeans = KMeans(n_clusters=k,init=X[final_medoids],max_iter=1)
        # kmeans = KMeans(n_clusters=k,init=centers,max_iter=3)
    kmeans.fit(X)
    labels=kmeans.labels_
    centers= kmeans.cluster_centers_
    inertia=kmeans.inertia_

    id=client_id
    centers=kmeans.cluster_centers_.tolist()
    labelss=kmeans.labels_.tolist()
    inertia=kmeans.inertia_

    receiver.sendResult(labelss,inertia,id)




# print(labelss)
# Get the initial data from the server
# initial_data = receiver.get_initial_data()

# # Add a random number to the initial data
# modified_data = initial_data + random.randint(1, 100)

# # Send the modified data back to the server
# print(client_id)
# receiver.set_modified_data(client_id, modified_data)
