import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib

with open('encoded_video_features.pickle', 'rb') as f:
    encoded_video_features = pickle.load(f)

with open('encoded_video_features_normal.pickle', 'rb') as f:
    encoded_video_features_normal = pickle.load(f)

train_collision_index_path = "/mnt/NAS/data/ruixuan/data/processed_WTS/train_collision_index.pkl"
with open(train_collision_index_path, 'rb') as f:
    train_collision_index = pickle.load(f)

x_train = np.array(list(encoded_video_features.values()))
x_train_normal = np.array(list(encoded_video_features_normal.values()))
y_train = np.array(train_collision_index)
y_train_normal = -1*np.ones(len(x_train_normal))
X = np.concatenate([x_train, x_train_normal], axis=0)
Y = np.concatenate([y_train, y_train_normal], axis=0)

X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[Y == 1, 0], X_embedded[Y == 1, 1], X_embedded[Y == 1, 2], c='r', marker='o', label='Collision')
ax.scatter(X_embedded[Y == 0, 0], X_embedded[Y == 0, 1], X_embedded[Y == 0, 2], c='g', marker='o', label='Near Miss')
ax.scatter(X_embedded[Y == -1, 0], X_embedded[Y == -1, 1], X_embedded[Y == -1, 2], c='b', marker='o', label='Normal')

# Set labels and title
ax.set_xlabel([])
ax.set_ylabel([])
ax.set_zlabel([])
plt.legend()
plt.show()