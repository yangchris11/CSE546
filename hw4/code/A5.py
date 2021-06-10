import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mnist import MNIST

sns.set()

np.random.seed(1968990 + 20210601)


def load_dataset():
    mnist_data = MNIST('./mnist/')
    X_train, Y_train = map(np.array, mnist_data.load_training())
    X_test, Y_test = map(np.array, mnist_data.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test


def k_means_obj(clusters):
    dist = 0
    cnt = 0
    for centroid, points in clusters.items():
        for point in points:
            dist += np.linalg.norm(centroid - point)
            cnt += 1
    return dist / cnt


class LloydsAlgorithm:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

        self.x_data, _, self.x_test, _ = load_dataset()
        self.n, self.d = self.x_data.shape
        self.current_iter = 0
        # self.distances = [float('inf')]
        # initial clustering
        self.clusters = { tuple(centroid): [] for centroid in self.x_data[np.random.choice(np.arange(self.n), self.k)]}
        self.test_clusters = {}
        self.clustering()
        self.distances = [k_means_obj(self.clusters)]
        self.centering()

    def clustering(self):
        centroids = [np.asarray(tuple_centroid) for tuple_centroid in self.clusters.keys()]
        for centroid in self.clusters.keys():
            self.clusters[centroid] = []
        for point in self.x_data:
            closest_centroid = centroids[np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])]
            if tuple(closest_centroid) not in self.clusters:
                raise RuntimeError
            self.clusters[tuple(closest_centroid)].append(point)

    def centering(self):
        new_clusters = {}
        for centroid, points in self.clusters.items():
            new_centroid = np.mean(points, axis=0)
            new_clusters[tuple(new_centroid)] = points
        self.clusters = new_clusters

    def test_clustering(self):
        centroids = [np.asarray(tuple_centroid) for tuple_centroid in self.clusters.keys()]
        for centroid in self.clusters.keys():
            self.test_clusters[centroid] = []
        for point in self.x_test:
            closest_centroid = centroids[np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])]
            if tuple(closest_centroid) not in self.test_clusters:
                raise RuntimeError
            self.test_clusters[tuple(closest_centroid)].append(point)

    def iter(self, threshold=1e-4):
        while self.current_iter <= self.max_iter:
            self.current_iter += 1
            start_time = time.time()
            old_dist = self.distances[-1]
            old_clusters = self.clusters

            self.clustering()
            self.distances.append(k_means_obj(self.clusters))
            self.centering()

            new_dist = self.distances[-1]
            # if (old_dist - new_dist) < threshold or (old_dist - new_dist) > (self.distances[-2] - old_dist):
            if old_dist < new_dist:
                self.clusters = old_clusters
                self.current_iter -= 1
                print('(k={}) Iter {}: {} (in {}s)'.format(self.k, self.current_iter, old_dist - new_dist, time.time() - start_time))
                break
            if (old_dist - new_dist) < threshold:
                print('(k={}) Iter {}: {} (in {}s)'.format(self.k, self.current_iter, old_dist - new_dist, time.time() - start_time))
                break
            else:
                print('(k={}) Iter {}: {} (in {}s)'.format(self.k, self.current_iter, old_dist - new_dist, time.time() - start_time))


def main():
    kmeans = LloydsAlgorithm(k=10, max_iter=10A0)
    kmeans.iter()

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, kmeans.current_iter + 1), kmeans.distances[1:])
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function')
    plt.savefig('A5b_obj.png')

    plt.figure(figsize=(10, 2))
    fig, axes = plt.subplots(1, 10)
    for i, (ax, centroid) in enumerate(zip(axes.ravel(), kmeans.clusters.keys())):
        ax.imshow(np.asarray(centroid).reshape((28, 28)))
        ax.axis('off')
    plt.savefig('A5b_centroid.png')
    plt.tight_layout()


def A5c():
    ks = [2, 4, 8, 16, 32, 64]
    train_objs = []
    test_objs = []
    for k in ks:
        kmeans = LloydsAlgorithm(k=k, max_iter=100)
        kmeans.iter()
        kmeans.clustering()
        kmeans.test_clustering()

        train_obj = k_means_obj(kmeans.clusters)
        test_obj = k_means_obj(kmeans.test_clusters)

        print('Training Objective: {} / Test Objective: {}'.format(train_obj, test_obj))

        train_objs.append(train_obj)
        test_objs.append(test_obj)

    plt.figure(figsize=(10, 6))
    plt.scatter(ks, train_objs, label='Training Error')
    plt.scatter(ks, train_objs, label='Test Error')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Objective Function')
    plt.savefig('A5c.png')


if __name__ == '__main__':
    main()
    A5c()