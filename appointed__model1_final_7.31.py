#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time


# In[16]:


appointed_original_feature=pd.read_csv("./appointed_model1_21_original_featureselection.csv")


# In[17]:


appointed_original_feature.head()


# In[18]:


appointed_original_feature = sm.add_constant(appointed_original_feature, has_constant='add')


# In[34]:


appointed_original_feature_X = appointed_original_feature.drop('target',axis=1)
appointed_original_feature_y = appointed_original_feature['target']

XX_train, XX_test, yy_train, yy_test = train_test_split(appointed_original_feature_X, appointed_original_feature_y, test_size=0.2,
                                                   random_state=100)
model1 = sm.Logit(yy_train,XX_train)
results=model1.fit(mothod='newton')
results.summary()


# In[35]:


np.exp(results.params)


# In[20]:


appointed_original_feature_X = appointed_original_feature.drop('target',axis=1)
appointed_original_feature_y = appointed_original_feature['target']


# In[21]:


from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit
import numpy as np

lr_clf = LogisticRegression(solver = 'newton-cg', multi_class='ovr', random_state = 100)

data = appointed_original_feature_X
label = appointed_original_feature_y


cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=100)
scores = cross_val_score(lr_clf, data, label, scoring = 'accuracy',cv = cv)
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))


# In[22]:


scores


# In[23]:


appointed_normal=pd.read_csv("./appointed_model1_21_normalization.csv")

appointed_normal_X = appointed_normal.drop('target',axis=1)
appointed_normal_y = appointed_normal['target']

X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(appointed_normal_X, appointed_normal_y, test_size=0.2, random_state=100)


# In[24]:


from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

data = appointed_normal_X
label = appointed_normal_y
cv_scores=[]
estimator_list = [i for i in range(0,50,1)]

for i in tqdm(range(0,50,1)):
    rfc = RandomForestClassifier(n_estimators = i+1, criterion='entropy', random_state = 100)
    score = cross_val_score(rfc, data, label, scoring = 'accuracy').mean()
    cv_scores.append(score)
    
best_e=[estimator_list[i] for i in range(len(cv_scores)) if cv_scores[i]==np.max(cv_scores)]
plt.figure(figsize=(20,10))
plt.legend(["Crodd validation scores"], fontsize=10)
plt.xlabel("the number of tree",fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.axvline(best_e[0], color='r',linestyle='--', linewidth=3)
plt.plot(estimator_list, cv_scores, marker='o', linestyle='dashed')
plt.show()


# In[25]:


print(f"최적의 tree개수:{(cv_scores.index(max(cv_scores)))+1}")


# In[26]:


a=cv_scores.index(max(cv_scores))+1


# In[27]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
select = RFE(RandomForestClassifier(n_estimators = a, random_state=100), n_features_to_select=3)

select.fit(X_normal_train,y_normal_train)
features_bool = np.array(select.get_support())
features = np.array(appointed_normal_X.columns)
result = features[features_bool]
print(result)


# In[28]:


appointed_normal_feature=pd.read_csv("./appointed_model1_21_normalization_featureselection.csv")

appointed_normal_feature_X = appointed_normal_feature.drop('target',axis=1)
appointed_normal_feature_y = appointed_normal_feature['target']


# In[29]:


from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

data = appointed_normal_feature_X
label = appointed_normal_feature_y
cv_scores=[]
estimator_list = [i for i in range(0,50,1)]

for i in tqdm(range(0,50,1)):
    rfc = RandomForestClassifier(n_estimators = i+1, criterion='entropy', random_state = 100)
    score = cross_val_score(rfc, data, label, scoring = 'accuracy').mean()
    cv_scores.append(score)
    
best_e=[estimator_list[i] for i in range(len(cv_scores)) if cv_scores[i]==np.max(cv_scores)]
plt.figure(figsize=(20,10))
plt.legend(["Crodd validation scores"], fontsize=10)
plt.xlabel("the number of tree",fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.axvline(best_e[0], color='r',linestyle='--', linewidth=3)
plt.plot(estimator_list, cv_scores, marker='o', linestyle='dashed')
plt.show()


# In[30]:


print(f"최적의 tree개수:{(cv_scores.index(max(cv_scores)))+1}")


# In[31]:


b=cv_scores.index(max(cv_scores))+1
b


# In[32]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = b, criterion='entropy',  random_state = 100) 
data = appointed_normal_feature_X
label = appointed_normal_feature_y

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
scores = cross_val_score(rf_clf, data, label, scoring = 'accuracy',cv = cv)
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))


# In[33]:


score


# In[ ]:





# In[ ]:




