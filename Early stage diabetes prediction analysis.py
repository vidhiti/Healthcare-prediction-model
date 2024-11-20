#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv('diabetes_data_upload.csv')
df.head()
df


# In[3]:


df.isna().sum()


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.columns = map(str.lower, df.columns)


# In[8]:


plt.figure(figsize=(8,6))
sns.distplot(df['age'],bins=30, color='royalblue')  
plt.title('Diabetes distribution by age', fontsize=17);


# In[9]:


sns.countplot(df['class'],data=df, palette=['#9080ff',"#ebdc78"])
plt.title('Diabetes positive & negative cases', fontsize=17);
plt.figure(figsize=(15,8))
plt.show()


# In[10]:


plt.title('Diabetes cases by Gender', fontsize=17);
sns.countplot(df['gender'],hue=df['class'], data=df, palette=['#9080ff',"#ebdc78"])


# In[11]:


plt.figure(figsize=(5,5))
plt.title('Diabetes positive & negative cases by age', fontsize=17);

ax = sns.violinplot(x="class", y="age", data=df, palette=['#22a7f0',"#e14b31"])
# ax.set_xticklabels({'class':['Negative','Positive']})
# ax.set_xticklabels([*'Positive'])


# In[12]:


plt.title('Diabetes cases by Gender', fontsize=17);
ax = sns.countplot(x="class", data=df, hue="gender", palette=['#22a7f0',"#e14b31"])


# In[13]:


df['gender'] = df['gender'].map({'Male':1,'Female':0})
df['class'] = df['class'].map({'Positive':1,'Negative':0})
df['polyuria'] = df['polyuria'].map({'Yes':1,'No':0})
df['polydipsia'] = df['polydipsia'].map({'Yes':1,'No':0})
df['sudden weight loss'] = df['sudden weight loss'].map({'Yes':1,'No':0})
df['weakness'] = df['weakness'].map({'Yes':1,'No':0})
df['polyphagia'] = df['polyphagia'].map({'Yes':1,'No':0})
df['genital thrush'] = df['genital thrush'].map({'Yes':1,'No':0})
df['visual blurring'] = df['visual blurring'].map({'Yes':1,'No':0})
df['itching'] = df['itching'].map({'Yes':1,'No':0})
df['irritability'] = df['irritability'].map({'Yes':1,'No':0})
df['delayed healing'] = df['delayed healing'].map({'Yes':1,'No':0})
df['partial paresis'] = df['partial paresis'].map({'Yes':1,'No':0})
df['muscle stiffness'] = df['muscle stiffness'].map({'Yes':1,'No':0})
df['alopecia'] = df['alopecia'].map({'Yes':1,'No':0})
df['obesity'] = df['obesity'].map({'Yes':1,'No':0})


# In[14]:


count = 1
plt.figure(figsize=(15,20))
plt.suptitle('Correlation between Symptoms and Diabetes cases'+ '\n', fontsize=20)
for i in df.columns:
    if i not in ['class', 'age', 'gender']:
        plt.subplot(5,4,count)
        plt.title(f'{i.title()}', fontweight='bold', fontsize=14)
        count +=1
        plt.tight_layout()
        df[i].value_counts().plot(kind="pie", colors=['royalblue','mediumpurple'],autopct='%1.1f%%',legend=True,labels=["Yes","No"])
        plt.ylabel('')
        plt.title(f'{i.title()}',fontweight='bold',fontsize=12)
        plt.legend(loc = "upper right",fontsize=12)
plt.tight_layout() 
plt.show()


# In[15]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True, cmap ='coolwarm')
plt.title('Correlation Heatmap\n',fontweight='bold',fontsize=14)
plt.show()


# # Pre-processing

# In[16]:


Y = df['class']
X = df.drop('class', axis=1)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=1)


# In[18]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[19]:


X = ss.fit_transform(X)


# In[20]:


X_train


# In[21]:


y_train


# # Model building

# In[22]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# # Linear Regression

# In[23]:


lin_model=LogisticRegression()
# lin_model = LogisticRegression(penalty='l2',C=1,max_iter=4)
lin_model.fit(X_train,y_train)
lin_pred = lin_model.predict(X_test)
linscore = lin_model.score(X_test,y_test)


# In[24]:


lin_error = mean_squared_error(y_test, lin_pred)
print("The Mean Squared Error For Linear Regression is: {}".format(lin_error))


# In[25]:


linscore = lin_model.score(X_test,y_test)
lin_cm = confusion_matrix(y_test,lin_pred)
lin_cr = classification_report(y_test,lin_pred)
print('Logistic Regression results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(linscore *100))
print('\n')
print('Confusion Matrix')
print(lin_cm)
print('\n')
print('Classification report')
print(lin_cr)                    


# # Decision Tree

# In[26]:


dtr_model = DecisionTreeClassifier(random_state=0)
dtr_model.fit(X_train,y_train)
dtr_pred = dtr_model.predict(X_test)


# In[27]:


dtr_error = mean_squared_error(y_test, dtr_pred)
print("The Mean Squared Error For Decision Tree Regression is: {}".format(dtr_error))


# In[28]:


dtscore = dtr_model.score(X_test,y_test)
dt_cm = confusion_matrix(y_test,dtr_pred)
dt_cr = classification_report(y_test,dtr_pred)
print('Decision Tree results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(dtscore *100))
print('\n')
print('Confusion Matrix')
print(dt_cm)
print('\n')
print('Classification report')
print(dt_cr)   


# # Linear SVM

# In[29]:


svm_model=SVC(kernel='linear',random_state=0)
svm_model.fit(X_train,y_train)
svm_pred = svm_model.predict(X_test)


# In[30]:


svm_error = mean_squared_error(y_test, svm_pred)
print("The Mean Squared Error For Support Vector Machine is: {}".format(svm_error))


# In[31]:


svmscore= svm_model.score(X_test, y_test)
svm_cm = confusion_matrix(y_test,svm_pred)
svm_cr = classification_report(y_test,svm_pred)
print('Support Vector Machine results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(svmscore *100))
print('\n')
print('Confusion Matrix')
print(svm_cm)
print('\n')
print('Classification report')
print(svm_cr) 


# # Naive Bayes classifier

# In[32]:


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)


# In[33]:


nb_error = mean_squared_error(y_test, nb_pred)
print("The Mean Squared Error For Naive Bayes classifier is: {}".format(nb_error))


# In[34]:


nbscore = nb_model.score(X_test,y_test)
nb_cm = confusion_matrix(y_test,nb_pred)
nb_cr = classification_report(y_test,nb_pred)
print('Naive Bayes classifier results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(nbscore *100))
print('\n')
print('Confusion Matrix')
print(nb_cm)
print('\n')
print('Classification report')
print(nb_cr) 


# # K-Nearest Neighbour

# In[35]:


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)


# In[36]:


knn_error = mean_squared_error(y_test, knn_pred)
print("The Mean Squared Error For K Nearest Neighbour is: {}".format(knn_error))


# In[37]:


knnscore = nb_model.score(X_test,y_test)
knn_cm = confusion_matrix(y_test,knn_pred)
knn_cr = classification_report(y_test,knn_pred)
print('K Nearest Neighbour results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(knnscore *100))
print('\n')
print('Confusion Matrix')
print(knn_cm)
print('\n')
print('Classification report')
print(knn_cr) 


# # Random Forest

# In[38]:


rf_model = RandomForestClassifier(n_estimators=15, random_state = 0)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)


# In[39]:


rf_error = mean_squared_error(y_test, rf_pred)
print("The Mean Squared Error For Random Forest is: {}".format(rf_error))


# In[40]:


rfscore = rf_model.score(X_test,y_test)
rf_cm = confusion_matrix(y_test,rf_pred)
rf_cr = classification_report(y_test,rf_pred)
print('Random Forest results')
print('---------------------------')
print('Accuracy is {:.2f}%'.format(rfscore *100))
print('\n')
print('Confusion Matrix')
print(rf_cm)
print('\n')
print('Classification report')
print(rf_cr) 


# # Prediction Results Overview

# In[41]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
models = {
"Linear Regression": LogisticRegression(),
"Decision Tree": DecisionTreeClassifier(),
"Linear Support Vector Machine": SVC(),
"Naive Bayes Classifier": GaussianNB(),
"K-Nearest Neighbour": KNeighborsClassifier(),
"Random Forest": RandomForestClassifier(),

}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))


# In[42]:


f, axes = plt.subplots(1, 7, figsize=(25, 7), sharey='row')
for i, (name, model) in enumerate(models.items()):
    y_pred = model.fit(X_train, y_train).predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cf_matrix)
    disp.plot(ax=axes[i], xticks_rotation=45, cmap='plasma')
    disp.ax_.set_title(name)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    if i!=0:
        disp.ax_.set_ylabel('')
f.text(0.4, 0.1, 'Predicted label', ha='left')
plt.subplots_adjust(wspace=0.40, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.show()


# In[ ]:




