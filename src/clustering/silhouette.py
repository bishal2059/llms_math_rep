from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_silhouette(embeddings, n_clusters: int = 4, seed: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    cluster_ids = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, cluster_ids)
    return score, cluster_ids