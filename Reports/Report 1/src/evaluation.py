"""External evaluation of a clustering against known class labels.

The labels are used ONLY here, after clustering, never by preprocessing or FCM.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def contingency(y, labels):
    """Clusters (rows) x classes (columns): how many points of each class fall in each cluster."""
    return pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y, name="class"))


def cluster_to_class(y, labels):
    """Mapping cluster -> class.

    If there are as many clusters as classes: the one-to-one mapping with the most agreement
    (all permutations are tried). Otherwise: each cluster gets its majority class.
    """
    tab = contingency(y, labels)
    clusters, classes = list(tab.index), list(tab.columns)
    if len(clusters) == len(classes):
        best = max(itertools.permutations(classes),
                   key=lambda perm: sum(tab.loc[k, cl] for k, cl in zip(clusters, perm)))
        return dict(zip(clusters, best))
    return tab.idxmax(axis=1).to_dict()


def external_scores(y, labels, mapping=None):
    """ARI, NMI, accuracy after mapping clusters to classes, and recall per class.

    ARI (Adjusted Rand Index): 0 = chance-level agreement, 1 = identical partitions.
    NMI (Normalized Mutual Information): 0 = independent, 1 = identical.
    Both ignore which cluster is called 0 or 1. Accuracy needs the cluster -> class mapping;
    pass a mapping learned elsewhere (e.g. on a training set) to evaluate new points with it.
    Returns (scores dict, mapping).
    """
    y = np.asarray(y)
    labels = np.asarray(labels)
    if mapping is None:
        mapping = cluster_to_class(y, labels)
    pred = np.array([mapping.get(l) for l in labels], dtype=object)
    scores = {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "accuracy": float((pred == y).mean()),
    }
    for cl in np.unique(y):
        scores[f"recall_{cl}"] = float((pred[y == cl] == cl).mean())
    return scores, mapping
