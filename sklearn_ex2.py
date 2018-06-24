import mglearn
import matplotlib.pyplot as plt
import numpy as np

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
print(mglearn.discrete_scatter(X[:, 0], X[:, 1], y))
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
print("X:",X)
print("y:",y)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

print("Breast cancer:")
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))

from sklearn.datasets import load_boston
boston = load_boston()
print("Date shape: {}".format(boston.data.shape))

X,y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
print("x_train: {}".format(X_train))
print("y_train: {}".format(y_train))

print("x_test: {}".format(X_test))
print("y_test: {}".format(y_test))

from sklearn.neighbors import  KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {}".format(clf.score(X_test,y_test)))

fig,axes=plt.subplots(1,3,figsize=(10,3))
print("ZIP:",zip([1,3,9],axes))
for n_neighbors,ax in zip([1,3,9],axes):
      print("n_neighbors",n_neighbors)
      print("ax",ax)
      clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
      mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
      mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
      ax.set_title("{} neighbor(s)".format(n_neighbors))
      ax.set_xlabel("feature0")
      ax.set_ylabel("feature1")
axes[0].legend(loc=3)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
training_accurary=[]
test_accurary=[]
#try n_neighbors from 1 to 10
neighbors_settings=range(1,11)
for n_neighbors in neighbors_settings:
      #build model
      clf=KNeighborsClassifier(n_neighbors=n_neighbors)
      clf.fit(X_train,y_train)
      #record training set accurary
      training_accurary.append(clf.score(X_train,y_train))
      test_accurary.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,training_accurary,label='training_accurary')
plt.plot(neighbors_settings,test_accurary,label='test_accurary')
plt.xlabel('n_neighbors')
plt.ylabel('Accurary')
plt.legend()
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=5)
plt.show()

from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)
#split the wave dataset into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)

print("test set predictions:\n {}".format(reg.predict(X_test)))
#print('test set predictions:\n {:.1f}'.format(reg.predict(X_test)))
print("Test set R^2:{:.2f}".format(reg.score(X_test,y_test)))

fix,axes = plt.subplots(1,3,figsize=(15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors,ax in zip([1,3,9],axes):
      reg = KNeighborsRegressor(n_neighbors=n_neighbors)
      reg.fit(X_train,y_train)
      ax.plot(line,reg.predict(line))
      ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
      ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)

      ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                  n_neighbors,reg.score(X_train,y_train),
                  reg.score(X_test,y_test)
            )
      )
      ax.set_xlabel("Feature")
      ax.set_ylabel("Target")
axes[0].legend(["Model predictions","Training data/target",
                "Test data/target"],loc="best")
plt.show()

mglearn.plots.plot_linear_regression_wave()
plt.show()

from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)
print("lr.coef_:%s"% lr.coef_)
print("lr.intercept_:%s"% lr.intercept_)
print("training set score_:%f"% lr.score(X_train,y_train))
print("test set score_:%f"% lr.score(X_test,y_test))


from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

lr = LinearRegression().fit(X_train,y_train)
print("lr.coef_:%s"% lr.coef_)
print("lr.intercept_:%s"% lr.intercept_)
print("boston training set score_:%f"% lr.score(X_train,y_train))
print("boston test set score_:%f"% lr.score(X_test,y_test))


from sklearn.linear_model import Ridge

ridge=Ridge().fit(X_train,y_train)
print("Ridge training set score: {:.2f}".format(ridge.score(X_train,y_train)))
print("Ridge test set score: {:.2f}".format(ridge.score(X_test,y_test)))

ridge10=Ridge(alpha=10).fit(X_train,y_train)
print("Ridge of alpha=10 training set score: {:.2f}".format(ridge10.score(X_train,y_train)))
print("Ridge of alpha=10 test set score: {:.2f}".format(ridge10.score(X_test,y_test)))

ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
print("Ridge of alpha=0.1 training set score: {:.2f}".format(ridge01.score(X_train,y_train)))
print("Ridge of alpha=0.1 test set score: {:.2f}".format(ridge01.score(X_test,y_test)))


plt.plot(ridge.coef_, 's',label="Ridge alpha=1")
plt.plot(ridge.coef_, '^',label="Ridge alpha=10")
plt.plot(ridge.coef_, 'v',label="Ridge alpha=0.1")

plt.plot(lr.coef_,'o',label="linearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient  magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-30,30)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()


from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train,y_train)
print("training set score: %f" % lasso.score(X_train,y_train))
print("test set score: %f" % lasso.score(X_test,y_test))
print("number of features used: %d" % np.sum(lasso.coef_!=0))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X,y = mglearn.datasets.make_forge()

fig,axes = plt.subplots(1,2,figsize=(10,3))
for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
      clf = model.fit(X,y)
      mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=.7)
      mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
      ax.set_title("{}".format(clf.__class__.__name__))
      ax.set_xlabel("Feature 0")
      ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()

mglearn.plots.plot_linear_svc_regularization()
plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
logreg = LogisticRegression().fit(X_train,y_train)
print("Training set score of Linear LogisticRegression: {:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score of Linear LogisticRegression: {:.3f}".format(logreg.score(X_test,y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train,y_train)
print("Training set score of Linear LogisticRegression100: {:.3f}".format(logreg100.score(X_train,y_train)))
print("Test set score of Linear LogisticRegression100: {:.3f}".format(logreg100.score(X_test,y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("Training set score of Linear LogisticRegression001: {:.3f}".format(logreg001.score(X_train,y_train)))
print("Test set score of Linear LogisticRegression001: {:.3f}".format(logreg001.score(X_test,y_test)))

plt.plot(logreg.coef_.T, 'o',label="C=1")
plt.plot(logreg100.coef_.T, '^',label="C=100")
plt.plot(logreg001.coef_.T, 'v',label="C=0.01")

#plt.plot(lr.coef_,'o',label="linearRegression")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient  magnitude")
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.legend()
plt.show()
