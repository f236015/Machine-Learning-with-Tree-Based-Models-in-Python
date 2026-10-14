# Import the built-in dataset available in sklearn datasets
from sklearn.datasets import load_iris
# Import Pandas
import pandas as pd
iris = load_iris() # creating iris object to access its functions
# Each observation represents one flower and 4 columns represents 4 measurements
# modify the below code to access only the petal length and width
X=iris.data
y=iris.target
# Convert to DataFrame
df = pd.DataFrame(data=X, columns=iris.feature_names)
# Add target column (species)
df['target'] = iris.target
# Print the first 5 rows
print (_______)
# by using the iris object print the following outputs
# 1. feature_names, 2. target, 3. target_names 4. type of data 5. type target
6. shape of data
# you can use type function to print the type of data and target

# your code here ------------------------
# train the decision tree classifier (Note: name your model variable 'clf' to
use the plotting function below)
# plot the tree structure
# make class and probability prediction for the following data [[5, 1.5]]
____________ # making class prediction
____________ # making class probablity prediction
# plot the decision boundry
from matplotlib.colors import ListedColormap
# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
path = os.path.join(fig_id + "." + fig_extension)
print("Saving figure", fig_id)
if tight_layout:
plt.tight_layout()
plt.savefig(path, format=fig_extension, dpi=resolution)
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True,
legend=False, plot_training=True):
x1s = np.linspace(axes[0], axes[1], 100)
x2s = np.linspace(axes[2], axes[3], 100)
x1, x2 = np.meshgrid(x1s, x2s)
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = clf.predict(X_new).reshape(x1.shape)
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
if not iris:
custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
if plot_training:
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
plt.axis(axes)
if iris:
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
else:

plt.xlabel(r"$x_1$", fontsize=18)
plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
if legend:
plt.legend(loc="lower right", fontsize=14)
plt.figure(figsize=(8, 4))
plot_decision_boundary(clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
save_fig("decision_tree_decision_boundaries_plot")
plt.show()
# accuracy of your model
