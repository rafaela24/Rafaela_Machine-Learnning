
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer


# In[2]:


cancer = load_breast_cancer()


# In[3]:


print("cancer.keys(): \n{}".format(cancer.keys()))


# In[4]:


print(cancer.data)


# In[5]:


from sklearn.datasets import load_breast_cancer


# In[6]:


breast_cancer = load_breast_cancer()


# In[7]:


X = breast_cancer.data


# In[8]:


y = breast_cancer.target


# In[9]:


print(X.shape)


# In[10]:


print(y.shape)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[13]:


print(knn)


# In[14]:


knn.fit(X, y)


# In[15]:


cancer.data [0]


# In[23]:


knn.predict([[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
       1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01]])


# In[24]:


X_new = [[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
       1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01], [1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
       1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01]]


# In[25]:


knn.predict(X_new)


# In[26]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[27]:


knn.fit(X, y)


# In[28]:


knn.predict(X_new)


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


logreg = LogisticRegression()


# In[31]:


logreg.fit(X, y)


# In[32]:


logreg.predict(X_new)


# In[33]:


y_pred = logreg.predict(X)


# In[34]:


from sklearn import metrics


# In[35]:


print(metrics.accuracy_score(y, y_pred))


# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[39]:


from sklearn.cross_validation import train_test_split


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[41]:


logreg = LogisticRegression()


# In[42]:


logreg.fit(X_train, y_train)


# In[43]:


y_pred = logreg.predict(X_test)


# In[44]:


print(metrics.accuracy_score(y_test, y_pred))


# In[45]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[46]:


knn.fit(X_train, y_train)


# In[47]:


y_pred = knn.predict(X_test)


# In[48]:


print(metrics.accuracy_score(y_test, y_pred))


# In[49]:


k_range = list(range(1, 26))


# In[50]:


scores = []


# In[51]:


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[52]:


import matplotlib.pyplot as plt


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


plt.plot(k_range, scores)


# In[55]:


plt.xlabel('Value of K for KNN')


# In[56]:


plt.ylabel('Testing Accuracy')

