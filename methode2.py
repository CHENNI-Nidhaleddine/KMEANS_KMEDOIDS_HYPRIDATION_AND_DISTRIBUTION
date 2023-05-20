from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score

# Define MPI communicator, rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define data variables
k = 3
data = pd.read_csv("data.csv")
X = data.values[:,2:-1]
X_split = np.array_split(X, size)
n_samples = data.shape[0]
n_features = data.shape[1]

# Define a function to compute the euclidean distance matrix
def compute_distance_matrix(centroids):
    distance_matrix = np.zeros((len(centroids), len(centroids)))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            distance_matrix[i,j] = np.linalg.norm(centroids[i] - centroids[j])
    return distance_matrix

def find_closest_centroids(data_points, centroids):
    # Calculate the Euclidean distance between each data point and each centroid
    closest_cents = []
    for x in data_points:
        dists = []
        for cent in centroids:
            distance = np.linalg.norm(x - cent)
            dists.append(distance)
        closest_cents.append(np.argmin(dists))
    return closest_cents

# Define a function to refine the centroids using k-medoids
def refine_centroids(centroids, k):
    distance_matrix = compute_distance_matrix(centroids)
    medoids_idx = np.random.choice(range(len(centroids)), size=k, replace=False)
    medoids = centroids[medoids_idx]
    labels = np.argmin(distance_matrix[:, medoids_idx], axis=1)
    for i in range(k):
        if i not in labels:
            j = np.argmax(np.sum(distance_matrix[:, i][:, np.newaxis] == distance_matrix[:, medoids_idx], axis=1))
            medoids_idx[i] = j
            medoids[i] = centroids[j]
            labels = np.argmin(distance_matrix[:, medoids_idx], axis=1)
    return medoids

# Master process
if rank == 0:
    # Initialize variables
    centroids = np.zeros((k, n_features))
    labels = np.zeros(n_samples, dtype=np.int32)

    # Perform K-means clustering on first data split
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(X_split[0])
    centroids = kmeans.cluster_centers_
    print(kmeans.cluster_centers_)
    labels = kmeans.labels_

    # Receive refined centroids and labels from slaves
    for i in range(1, size):
        refined_centroids = comm.recv(source=i, tag=1)
        labels_chunk = comm.recv(source=i, tag=2)
        centroids = np.vstack((centroids, refined_centroids))
        labels = np.hstack((labels, labels_chunk))

    # Calculate final refined centroids
    refined_centroids = refine_centroids(centroids, k)

    # Broadcast final refined centroids to all slaves
    for i in range(1, size):
        comm.send(refined_centroids, dest=i, tag=3)

    # Calculate final labels and silhouette score
    kmeans = KMeans(n_clusters=k, init=refined_centroids, max_iter=100).fit(X)
    labels = kmeans.labels_
    inertia=kmeans.inertia_
    silouhet = silhouette_score(X, labels)

    # Print final results
    print("Final clustering labels:")
    print(labels)
    print("Final centroids:")
    print(refined_centroids)
    print("Silhouette score:")
    print(silouhet)
    print("NMI:")
    nmi = normalized_mutual_info_score(data.iloc[:,1], labels)    
    print(nmi)
    print("inertia:")
    print(inertia)
# Slave processes
else:
    # Perform K-means clustering on data split
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(X_split[rank])
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Send refined centroids and labels to master
    refined_centroids = refine_centroids(centroids, k)
    comm.send(refined_centroids, dest=0, tag=1)
    comm.send(labels, dest=0, tag=2)

    # Receive final refined centroids from master
    final_refined_centroids = comm.recv(source=0, tag=3)

    # Perform K-means clustering with final refined centroids
    kmeans = KMeans(n_clusters=k, init=final_refined_centroids, max_iter=100).fit(X_split[rank])
    labels = kmeans.labels_

    # Send final labels to master
    comm.send(labels, dest=0, tag=4)
