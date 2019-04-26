import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ga_nn import GBMLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("adsdata.csv")

df.drop(["UserID", "Gender"], axis=1, inplace=True)
df[["Age", "EstimatedSalary"]] -= df[["Age", "EstimatedSalary"]].mean()
# df[["Age", "EstimatedSalary"]] /= df[["Age", "EstimatedSalary"]].std()
# print(df.head(10))


# train and test split
x_t, x_v, y_t, y_v = train_test_split(df[["Age", "EstimatedSalary"]], df["Purchased"], test_size=0.3)

train = x_t.values
target = y_t.values
test = x_v.values
output = y_v.values

target.resize((target.shape[0], 1))

nn = GBMLPClassifier(hidden_layer=(10, 5,), population_size=100, gene_length=7, max_generation=100)

nn.fit(train, target)

y = nn.predict(test)
print("confusion matrix: ", confusion_matrix(output, y))
print("accuracy score: ", accuracy_score(output, y))

'''
# plotting graph
c = ["r", "g"]
fig = plt.figure()
for i in range(2):
    cl = df[df["Purchased"] == i]
    plt.scatter(cl["Age"], cl["EstimatedSalary"], c=c[i], marker="^")
plt.xlabel("Age")
plt.ylabel("EstimatedSalary")
plt.show()
'''

fig2 = plt.figure()
X_set, y_set = train, target
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 0.05, stop=X_set[:, 0].max() + 0.05, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 0.05, stop=X_set[:, 1].max() + 0.05, step=0.1))
space = np.array([X1.ravel(), X2.ravel()]).T
pred = nn.predict(space).reshape(X1.shape)
plt.contourf(X1, X2, pred, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
y_set.resize((y_set.shape[0]))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Genetic Neural net(Training set) (10, 5)')
plt.text(0.75, -0.05, "Accuracy score: "+str(round(accuracy_score(output, y), 4)), fontsize=10)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
