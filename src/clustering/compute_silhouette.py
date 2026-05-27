from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def compute_layer_silhouette(embeddings, n_clusters=4):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    score = silhouette_score(
        embeddings,
        cluster_labels
    )

    return score