#!/usr/bin/env python
# coding: utf-8

# # Dongkun
# # 1 Data Preparation

# In[1]:


# installing 1.0.3 because this version of pandas supports write to s3
get_ipython().system('pip install pandas==1.0.3')


# In[2]:


# This path will be active after the launch of the hackathon / day-day-up-unsw / s3://tf-trachack-notebooks/day-day-up-unsw/jupyter/jovyan/
teamname = 'day-day-up-unsw' #'trachack-a-groups-admin-py-tracfone'
data_folder='s3://tf-trachack-data/212/'
# change root_folder to your team's root folder
# s3://tf-trachack-notebooks/<this should be replaced by team name as provided in EMAIL>/jupyter/jovyan/
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'


# In[3]:


# import necessary library
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier    
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime


# # Data Preprocessing (Hanxi Liu)

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime
from pandas import to_datetime

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# In[5]:


upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
phone_info_eval=pd.read_csv(data_folder+"data/eval/phone_info.csv")
customer_info_eval=pd.read_csv(data_folder+"data/eval/customer_info.csv")
redemptions_eval=pd.read_csv(data_folder+"data/eval/redemptions.csv")
deactivations_eval=pd.read_csv(data_folder+"data/eval/deactivations.csv")
reactivations_eval=pd.read_csv(data_folder+"data/eval/reactivations.csv")
suspensions_eval=pd.read_csv(data_folder+"data/eval/suspensions.csv")
# network_usage_domestic_eval=pd.read_csv(data_folder+"data/eval/network_usage_domestic.csv")
# lrp_points_eval=pd.read_csv(data_folder+"data/eval/lrp_points.csv")
# lrp_enrollment_eval=pd.read_csv(data_folder+"data/eval/lrp_enrollment.csv")


# In[6]:


upgrade_phone_info_eval = pd.merge(upgrades_eval, phone_info_eval, on = 'line_id', how = 'inner')
upgrade_customer_info_eval = pd.merge(upgrades_eval, customer_info_eval, on = 'line_id', how = 'inner')
upgrade_redemptions_eval = pd.merge(upgrades_eval, redemptions_eval, on = 'line_id', how = 'inner')
upgrade_deactivations_eval = pd.merge(upgrades_eval, deactivations_eval, on = 'line_id', how = 'inner')
upgrade_reactivations_eval = pd.merge(upgrades_eval, reactivations_eval, on = 'line_id', how = 'inner')
upgrade_suspensions_eval = pd.merge(upgrades_eval, suspensions_eval, on = 'line_id', how = 'left')
# upgrade_network_usage_domestic_eval = pd.merge(upgrades_eval, network_usage_domestic_eval, on = 'line_id', how = 'inner')
# upgrade_lrp_points_eval = pd.merge(upgrades_eval, lrp_points_eval, on = 'line_id', how = 'left')
# upgrade_lrp_enrollment_eval = pd.merge(upgrades_eval, lrp_enrollment_eval, on = 'line_id', how = 'left')


# In[7]:


upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
phone_info=pd.read_csv(data_folder+"data/dev/phone_info.csv")
customer_info=pd.read_csv(data_folder+"data/dev/customer_info.csv")
redemptions=pd.read_csv(data_folder+"data/dev/redemptions.csv")
deactivations=pd.read_csv(data_folder+"data/dev/deactivations.csv")
reactivations=pd.read_csv(data_folder+"data/dev/reactivations.csv")
suspensions=pd.read_csv(data_folder+"data/dev/suspensions.csv")
# network_usage_domestic=pd.read_csv(data_folder+"data/dev/network_usage_domestic.csv")
# lrp_points=pd.read_csv(data_folder+"data/dev/lrp_points.csv")
# lrp_enrollment=pd.read_csv(data_folder+"data/dev/lrp_enrollment.csv")


# In[8]:


upgrade_phone_info = pd.merge(upgrades, phone_info, on = 'line_id', how = 'inner')
upgrade_customer_info = pd.merge(upgrades, customer_info, on = 'line_id', how = 'inner')
upgrade_redemptions = pd.merge(upgrades, redemptions, on = 'line_id', how = 'inner')
upgrade_deactivations = pd.merge(upgrades, deactivations, on = 'line_id', how = 'inner')
upgrade_reactivations = pd.merge(upgrades, reactivations, on = 'line_id', how = 'inner')
upgrade_suspensions = pd.merge(upgrades, suspensions, on = 'line_id', how = 'left')
# upgrade_network_usage_domestic = pd.merge(upgrades, network_usage_domestic, on = 'line_id', how = 'inner')
# upgrade_lrp_points = pd.merge(upgrades, lrp_points, on = 'line_id', how = 'left')
# upgrade_lrp_enrollment = pd.merge(upgrades, lrp_enrollment, on = 'line_id', how = 'left')


# In[9]:


# upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
# upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
# # upgrades.to_csv(data_folder+"code/data/dev/upgrades_new.csv",header=True,index=None)
# # upgrades_eval.to_csv(data_folder+"code/data/eval/upgrades_new_eval.csv",header=True,index=None)


# In[10]:


# def upgrade_phone_info_update():
 
#     for field in fields:
#         print(field)
#         upgrade_phone_info.loc[upgrade_phone_info[field].isnull(), field] = 'N'
#         upgrade_phone_info_eval.loc[upgrade_phone_info_eval[field].isnull(), field] = 'N'

# upgrade redemptions
# date_observed has only five kind of dates
# upgrade_redemptions.loc[upgrade_redemptions['date_observed'] == '2021-03-10', 'date_observed'] = 1
# upgrade_redemptions.loc[upgrade_redemptions['date_observed'] == '2021-03-11', 'date_observed'] = 2
# upgrade_redemptions.loc[upgrade_redemptions['date_observed'] == '2021-03-12', 'date_observed'] = 3
# upgrade_redemptions.loc[upgrade_redemptions['date_observed'] == '2021-03-13', 'date_observed'] = 4
# upgrade_redemptions.loc[upgrade_redemptions['date_observed'] == '2021-03-14', 'date_observed'] = 5

# upgrade_redemptions_eval.loc[upgrade_redemptions_eval['date_observed'] == '2021-03-10', 'date_observed'] = 1
# upgrade_redemptions_eval.loc[upgrade_redemptions_eval['date_observed'] == '2021-03-11', 'date_observed'] = 2
# upgrade_redemptions_eval.loc[upgrade_redemptions_eval['date_observed'] == '2021-03-12', 'date_observed'] = 3
# upgrade_redemptions_eval.loc[upgrade_redemptions_eval['date_observed'] == '2021-03-13', 'date_observed'] = 4
# upgrade_redemptions_eval.loc[upgrade_redemptions_eval['date_observed'] == '2021-03-14', 'date_observed'] = 5

# redemption_date - Do standardization processing for month, year and day respectively
upgrade_redemptions.loc[:,'redemption_year'] = upgrade_redemptions['redemption_date'].apply(lambda x:x.split('-')[0])
upgrade_redemptions.loc[:,'redemption_month'] = upgrade_redemptions['redemption_date'].apply(lambda x:x.split('-')[1])
upgrade_redemptions.loc[:,'redemption_day'] = upgrade_redemptions['redemption_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_redemptions.loc[:,['redemption_year', 'redemption_month', 'redemption_day']])
upgrade_redemptions.loc[:,'redemption_year'] = data[:, 0]
upgrade_redemptions.loc[:,'redemption_month'] = data[:, 1]
upgrade_redemptions.loc[:,'redemption_day'] = data[:, 2]

upgrade_redemptions_eval.loc[:,'redemption_year'] = upgrade_redemptions_eval['redemption_date'].apply(lambda x:x.split('-')[0])
upgrade_redemptions_eval.loc[:,'redemption_month'] = upgrade_redemptions_eval['redemption_date'].apply(lambda x:x.split('-')[1])
upgrade_redemptions_eval.loc[:,'redemption_day'] = upgrade_redemptions_eval['redemption_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_redemptions_eval.loc[:,['redemption_year', 'redemption_month', 'redemption_day']])
upgrade_redemptions_eval.loc[:,'redemption_year'] = data[:, 0]
upgrade_redemptions_eval.loc[:,'redemption_month'] = data[:, 1]
upgrade_redemptions_eval.loc[:,'redemption_day'] = data[:, 2]

# channel, redemption_type, revenue_type
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_redemptions.loc[:, ['revenue_type']])

n_samples = contents.shape[1]
redemptions_field = []
for i in range(n_samples):
    s = 'redemptions_' + str(i)
    redemptions_field.append(s)
    
for i in range(n_samples):
    upgrade_redemptions[redemptions_field[i]] = contents[:, i]

#Eavlation data set
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_redemptions_eval.loc[:, ['revenue_type']])

n_samples = contents.shape[1]
redemptions_field = []
for i in range(n_samples):
    s = 'redemptions_' + str(i)
    redemptions_field.append(s)
    
for i in range(n_samples):
    upgrade_redemptions_eval[redemptions_field[i]] = contents[:, i]

# gross_revenue, add column ave_gross_revenue
upgrade_redemptions['ave_gross_revenue'] = upgrade_redemptions.groupby('line_id')['gross_revenue'].transform('mean')
st=StandardScaler() # normalize
data=st.fit_transform(upgrade_redemptions.loc[:,['ave_gross_revenue']])
upgrade_redemptions['ave_gross_revenue'] = data

upgrade_redemptions_eval['ave_gross_revenue'] = upgrade_redemptions_eval.groupby('line_id')['gross_revenue'].transform('mean')
st=StandardScaler()
data=st.fit_transform(upgrade_redemptions_eval.loc[:,['ave_gross_revenue']])
upgrade_redemptions_eval['ave_gross_revenue'] = data
# redemption_count, record the number of redemptions for each line_id
# upgrade_redemptions['redemption_count'] = upgrade_redemptions.groupby('line_id')['line_id'].transform('count')

train_set_path=root_folder+"code/data/dev/redemptions_new.csv"
upgrade_redemptions.to_csv(train_set_path, header=True, index=None)

eval_set_path=root_folder+"code/data/eval/redemptions_new_eval.csv"
upgrade_redemptions_eval.to_csv(eval_set_path, header=True, index=None)


# In[11]:


upgrade_redemptions = upgrade_redemptions.drop(['channel', 'redemption_date', 'redemption_type', 'revenue_type', 'date_observed','gross_revenue'],axis=1)
upgrade_redemptions


# In[12]:


# agg with maximum value according to redemption_year
upgrade_redemptions = upgrade_redemptions.groupby('line_id').agg(['max']).reset_index()


# ### deactivations

# In[13]:


# deactivation_date - Do one hot processing for month, year and day respectively
upgrade_deactivations.loc[:,'deactivation_year'] = upgrade_deactivations['deactivation_date'].apply(lambda x:x.split('-')[0])
upgrade_deactivations.loc[:,'deactivation_month'] = upgrade_deactivations['deactivation_date'].apply(lambda x:x.split('-')[1])
upgrade_deactivations.loc[:,'deactivation_day'] = upgrade_deactivations['deactivation_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_deactivations.loc[:,['deactivation_year', 'deactivation_month', 'deactivation_day']])
upgrade_deactivations.loc[:,'deactivation_year'] = data[:, 0]
upgrade_deactivations.loc[:,'deactivation_month'] = data[:, 1]
upgrade_deactivations.loc[:,'deactivation_day'] = data[:, 2]

upgrade_deactivations_eval.loc[:,'deactivation_year'] = upgrade_deactivations_eval['deactivation_date'].apply(lambda x:x.split('-')[0])
upgrade_deactivations_eval.loc[:,'deactivation_month'] = upgrade_deactivations_eval['deactivation_date'].apply(lambda x:x.split('-')[1])
upgrade_deactivations_eval.loc[:,'deactivation_day'] = upgrade_deactivations_eval['deactivation_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_deactivations_eval.loc[:,['deactivation_year', 'deactivation_month', 'deactivation_day']])
upgrade_deactivations_eval.loc[:,'deactivation_year'] = data[:, 0]
upgrade_deactivations_eval.loc[:,'deactivation_month'] = data[:, 1]
upgrade_deactivations_eval.loc[:,'deactivation_day'] = data[:, 2]

train_set_path=root_folder+"code/data/dev/deactivations_new.csv"
upgrade_deactivations.to_csv(train_set_path,header=True,index=None)

eval_set_path=root_folder+"code/data/eval/deactivations_new_eval.csv"
upgrade_deactivations_eval.to_csv(eval_set_path,header=True,index=None)


# ### reactivations

# In[14]:


# reactivation_date - Do one hot processing for month, year and day respectively
upgrade_reactivations.loc[:,'reactivation_year'] = upgrade_reactivations['reactivation_date'].apply(lambda x:x.split('-')[0])
upgrade_reactivations.loc[:,'reactivation_month'] = upgrade_reactivations['reactivation_date'].apply(lambda x:x.split('-')[1])
upgrade_reactivations.loc[:,'reactivation_day'] = upgrade_reactivations['reactivation_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_reactivations.loc[:,['reactivation_year', 'reactivation_month', 'reactivation_day']])
upgrade_reactivations.loc[:,'reactivation_year'] = data[:, 0]
upgrade_reactivations.loc[:,'reactivation_month'] = data[:, 1]
upgrade_reactivations.loc[:,'reactivation_day'] = data[:, 2]

upgrade_reactivations_eval.loc[:,'reactivation_year'] = upgrade_reactivations_eval['reactivation_date'].apply(lambda x:x.split('-')[0])
upgrade_reactivations_eval.loc[:,'reactivation_month'] = upgrade_reactivations_eval['reactivation_date'].apply(lambda x:x.split('-')[1])
upgrade_reactivations_eval.loc[:,'reactivation_day'] = upgrade_reactivations_eval['reactivation_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_reactivations_eval.loc[:,['reactivation_year', 'reactivation_month', 'reactivation_day']])
upgrade_reactivations_eval.loc[:,'reactivation_year'] = data[:, 0]
upgrade_reactivations_eval.loc[:,'reactivation_month'] = data[:, 1]
upgrade_reactivations_eval.loc[:,'reactivation_day'] = data[:, 2]

train_set_path=root_folder+"code/data/dev/reactivations_new.csv"
upgrade_reactivations.to_csv(train_set_path,header=True,index=None)

eval_set_path=root_folder+"code/data/eval/reactivations_new_eval.csv"
upgrade_reactivations_eval.to_csv(eval_set_path,header=True,index=None)


# ### suspensions

# In[15]:


# Drop null field in upgrade_suspensions
upgrade_suspensions = upgrade_suspensions.dropna()
upgrade_suspensions_eval = upgrade_suspensions_eval.dropna()

# suspension_start_date - Do one hot processing for month, year and day respectively
upgrade_suspensions.loc[:,'suspension_start_year'] = upgrade_suspensions['suspension_start_date'].apply(lambda x:x.split('-')[0])
upgrade_suspensions.loc[:,'suspension_start_month'] = upgrade_suspensions['suspension_start_date'].apply(lambda x:x.split('-')[1])
upgrade_suspensions.loc[:,'suspension_start_day'] = upgrade_suspensions['suspension_start_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_suspensions.loc[:,['suspension_start_year', 'suspension_start_month', 'suspension_start_day']])
upgrade_suspensions.loc[:,'suspension_start_year'] = data[:, 0]
upgrade_suspensions.loc[:,'suspension_start_month'] = data[:, 1]
upgrade_suspensions.loc[:,'suspension_start_day'] = data[:, 2]

upgrade_suspensions_eval.loc[:,'suspension_start_year'] = upgrade_suspensions_eval['suspension_start_date'].apply(lambda x:x.split('-')[0])
upgrade_suspensions_eval.loc[:,'suspension_start_month'] = upgrade_suspensions_eval['suspension_start_date'].apply(lambda x:x.split('-')[1])
upgrade_suspensions_eval.loc[:,'suspension_start_day'] = upgrade_suspensions_eval['suspension_start_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_suspensions_eval.loc[:,['suspension_start_year', 'suspension_start_month', 'suspension_start_day']])
upgrade_suspensions_eval.loc[:,'suspension_start_year'] = data[:, 0]
upgrade_suspensions_eval.loc[:,'suspension_start_month'] = data[:, 1]
upgrade_suspensions_eval.loc[:,'suspension_start_day'] = data[:, 2]

# suspension_end_date - Do one hot processing for month, year and day respectively
upgrade_suspensions.loc[:,'suspension_end_year'] = upgrade_suspensions['suspension_end_date'].apply(lambda x:x.split('-')[0])
upgrade_suspensions.loc[:,'suspension_end_month'] = upgrade_suspensions['suspension_end_date'].apply(lambda x:x.split('-')[1])
upgrade_suspensions.loc[:,'suspension_end_day'] = upgrade_suspensions['suspension_end_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_suspensions.loc[:,['suspension_end_year', 'suspension_end_month', 'suspension_end_day']])
upgrade_suspensions.loc[:,'suspension_end_year'] = data[:, 0]
upgrade_suspensions.loc[:,'suspension_end_month'] = data[:, 1]
upgrade_suspensions.loc[:,'suspension_end_day'] = data[:, 2]

upgrade_suspensions_eval.loc[:,'suspension_end_year'] = upgrade_suspensions_eval['suspension_end_date'].apply(lambda x:x.split('-')[0])
upgrade_suspensions_eval.loc[:,'suspension_end_month'] = upgrade_suspensions_eval['suspension_end_date'].apply(lambda x:x.split('-')[1])
upgrade_suspensions_eval.loc[:,'suspension_end_day'] = upgrade_suspensions_eval['suspension_end_date'].apply(lambda x:x.split('-')[2])
st=StandardScaler()
data=st.fit_transform(upgrade_suspensions_eval.loc[:,['suspension_end_year', 'suspension_end_month', 'suspension_end_day']])
upgrade_suspensions_eval.loc[:,'suspension_end_year'] = data[:, 0]
upgrade_suspensions_eval.loc[:,'suspension_end_month'] = data[:, 1]
upgrade_suspensions_eval.loc[:,'suspension_end_day'] = data[:, 2]

train_set_path=root_folder+"code/data/dev/suspensions_new.csv"
upgrade_suspensions.to_csv(train_set_path,header=True,index=None)

eval_set_path=root_folder+"code/data/eval/suspensions_new_eval.csv"
upgrade_suspensions_eval.to_csv(eval_set_path,header=True,index=None)


# # merge table and generate new train set and eval set

# In[16]:


data_folder=root_folder
# upgrades=pd.read_csv(data_folder+"code/data/dev/upgrades_new.csv")
# phone_info=pd.read_csv(data_folder+"code/data/dev/phone_info.csv")
# customer_info=pd.read_csv(data_folder+"code/data/dev/customer_info.csv")
redemptions=pd.read_csv(data_folder+"code/data/dev/redemptions_new.csv")
deactivations=pd.read_csv(data_folder+"code/data/dev/deactivations_new.csv")
reactivations=pd.read_csv(data_folder+"code/data/dev/reactivations_new.csv")
suspensions=pd.read_csv(data_folder+"code/data/dev/suspensions_new.csv")
# network_usage_domestic=pd.read_csv(data_folder+"code/data/dev/network_usage_domestic.csv")
# lrp_points=pd.read_csv(data_folder+"code/data/dev/lrp_points.csv")
# lrp_enrollment=pd.read_csv(data_folder+"code/data/dev/lrp_enrollment.csv")


# In[17]:


# upgrade_phone_info = pd.merge(upgrades, phone_info, on = 'line_id', how = 'inner')
# upgrade_customer_info = pd.merge(upgrades, customer_info, on = 'line_id', how = 'inner')
# upgrade_redemptions = pd.merge(upgrades, redemptions, on = 'line_id', how = 'inner')
# upgrade_deactivations = pd.merge(upgrades, deactivations, on = 'line_id', how = 'inner')
# upgrade_reactivations = pd.merge(upgrades, reactivations, on = 'line_id', how = 'inner')
# upgrade_suspensions = pd.merge(upgrades, suspensions, on = 'line_id', how = 'left')
# upgrade_network_usage_domestic = pd.merge(upgrades, network_usage_domestic, on = 'line_id', how = 'inner')
# upgrade_lrp_points = pd.merge(upgrades, lrp_points, on = 'line_id', how = 'left')
# upgrade_lrp_enrollment = pd.merge(upgrades, lrp_enrollment, on = 'line_id', how = 'left')


# In[18]:


# upgrades_eval=pd.read_csv(data_folder+"code/data/eval/upgrades_new_eval.csv")
# phone_info_eval=pd.read_csv(data_folder+"code/data/eval/phone_info.csv")
# customer_info_eval=pd.read_csv(data_folder+"code/data/eval/customer_info.csv")
redemptions_eval=pd.read_csv(data_folder+"code/data/eval/redemptions_new_eval.csv")
deactivations_eval=pd.read_csv(data_folder+"code/data/eval/deactivations_new_eval.csv")
reactivations_eval=pd.read_csv(data_folder+"code/data/eval/reactivations_new_eval.csv")
suspensions_eval=pd.read_csv(data_folder+"code/data/eval/suspensions_new_eval.csv")
# network_usage_domestic_eval=pd.read_csv(data_folder+"code/data/eval/network_usage_domestic.csv")
# lrp_points_eval=pd.read_csv(data_folder+"code/data/eval/lrp_points.csv")
# lrp_enrollment_eval=pd.read_csv(data_folder+"code/data/eval/lrp_enrollment.csv")


# In[19]:


redemptions = redemptions.drop(columns=['date_observed', 'channel', 'gross_revenue', 'redemption_date', 'redemption_type', 'revenue_type', 'upgrade'])
redemptions_eval = redemptions_eval.drop(columns=['date_observed', 'channel', 'gross_revenue', 'redemption_date', 'redemption_type', 'revenue_type'])

#聚合处理
redemptions = redemptions.groupby('line_id').max()
redemptions_eval = redemptions_eval.groupby('line_id').max()
redemptions.head(5)


# In[20]:


deactivations = deactivations.drop(columns=['date_observed', 'deactivation_date', 'deactivation_reason', 'upgrade'])
deactivations_eval = deactivations_eval.drop(columns=['date_observed', 'deactivation_date', 'deactivation_reason'])
# 聚合处理
deactivations = deactivations.groupby('line_id').max()
deactivations_eval = deactivations_eval.groupby('line_id').max()
deactivations.head(5)


# In[21]:


reactivations = reactivations.drop(columns=['date_observed', 'reactivation_channel', 'reactivation_date','upgrade'])
reactivations_eval = reactivations_eval.drop(columns=['date_observed', 'reactivation_channel', 'reactivation_date'])
# 聚合处理
reactivations = reactivations.groupby('line_id').max()
reactivations_eval = reactivations_eval.groupby('line_id').max()
reactivations.head(5)


# In[22]:


suspensions = suspensions.drop(columns=['date_observed', 'suspension_start_date', 'suspension_end_date', 'upgrade'])
suspensions_eval = suspensions_eval.drop(columns=['date_observed', 'suspension_start_date', 'suspension_end_date'])
# 聚合处理
suspensions = suspensions.groupby('line_id').max()
suspensions_eval = suspensions_eval.groupby('line_id').max()
suspensions.head(5)


# In[23]:


# Train data set
upgrade_redemptions = pd.merge(upgrades, redemptions, on ='line_id', how = 'left')
upgrade_redemptions_deactivations = pd.merge(upgrade_redemptions, deactivations, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_reactivations = pd.merge(upgrade_redemptions_deactivations, reactivations, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_reactivations_suspensions = pd.merge(upgrade_redemptions_deactivations_reactivations, suspensions, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_reactivations_suspensions.head(5)


# In[24]:


# Define train set
train_set = upgrade_redemptions_deactivations_reactivations_suspensions
# 清理未处理的数据
train_set = train_set.drop(columns=['date_observed'])
# 处理 upgrade 列
train_set.loc[train_set.upgrade == 'yes', 'upgrade'] = 1
train_set.loc[train_set.upgrade == 'no', 'upgrade'] = 0

upgrade_redemptions_deactivations_reactivations_suspensions.head(5)


# In[25]:


# 缺失值处理
data = train_set.iloc[:, 1:]
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
data
train_set.iloc[:, 1:] = data


# In[26]:


# Eval data set
upgrade_redemptions_eval = pd.merge(upgrades_eval, redemptions_eval, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_eval = pd.merge(upgrade_redemptions_eval, deactivations_eval, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_reactivations_eval = pd.merge(upgrade_redemptions_deactivations_eval, reactivations_eval, on = 'line_id', how = 'left')
upgrade_redemptions_deactivations_reactivations_suspensions_eval = pd.merge(upgrade_redemptions_deactivations_reactivations_eval, suspensions_eval, on = 'line_id', how = 'left')


# In[27]:


# Define eval set
eval_set = upgrade_redemptions_deactivations_reactivations_suspensions_eval
# 清理未处理的数据
eval_set = eval_set.drop(columns=['date_observed'])

upgrade_redemptions_deactivations_reactivations_suspensions_eval.head(5)


# In[28]:


# 缺失值处理
data = eval_set.iloc[:, 1:]
print(data.shape)
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
eval_set.iloc[:, 1:] = data
print(data.shape)
# 这里transform之后数据维度降低了
# columns = eval_set.columns
# eval_set.drop(columns=columns[1:])
# df = pd.concat([eval_set.iloc[:,0], pd.DataFrame(data)])


# In[29]:


# train_set = train_set.loc[:, fields]
# eval_set = eval_set.loc[:, fields]
# train_set.to_csv(root_folder+"Hanxi Liu/train_set.csv", header=True,index=None)
# eval_set.to_csv(root_folder+"Hanxi Liu/eval_set.csv", header=True,index=None)
train_set.to_csv(root_folder+"code/data/dev/upgrade_redemptions_deactivations_reactivations_suspensions.csv", header=True,index=None)
eval_set.to_csv(root_folder+"code/data/eval/upgrade_redemptions_deactivations_reactivations_suspensions_eval.csv", header=True,index=None)
train_set


# ### Customer info

# In[30]:


# This path will be active after the launch of the hackathon / day-day-up-unsw / s3://tf-trachack-notebooks/day-day-up-unsw/jupyter/jovyan/
teamname = 'day-day-up-unsw' #'trachack-a-groups-admin-py-tracfone'
data_folder='s3://tf-trachack-data/212/'
# change root_folder to your team's root folder
# s3://tf-trachack-notebooks/<this should be replaced by team name as provided in EMAIL>/jupyter/jovyan/
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'


# In[31]:


upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
customer_info=pd.read_csv(data_folder+"data/dev/customer_info.csv")


# In[32]:


upgrade_customer_info = pd.merge(upgrades, customer_info, on = 'line_id', how = 'inner')


# In[33]:


upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
customer_info_eval=pd.read_csv(data_folder+"data/eval/customer_info.csv")


# In[34]:


upgrade_customer_info_eval = pd.merge(upgrades_eval, customer_info_eval, on = 'line_id', how = 'inner')


# In[35]:


# upgrade_customer_info[upgrade_customer_info.loc[:, 'upgrade'] == 'yes'] = 1
# upgrade_customer_info[upgrade_customer_info.loc[:, 'upgrade'] == 'no'] = 0


# In[36]:


#Train
upgrade_customer_info.loc[:, 'first_activation_date'].fillna("0-0-0", inplace=True)
data = upgrade_customer_info.loc[:, 'first_activation_date']
#Eval
upgrade_customer_info_eval.loc[:, 'first_activation_date'].fillna("0-0-0", inplace=True)
data = upgrade_customer_info_eval.loc[:, 'first_activation_date']


# In[37]:


#Train
data = upgrade_customer_info.loc[:, 'first_activation_date']
upgrade_customer_info['first_activation_year'] = data.apply(lambda x:str(x).split('-')[0])

#Eval
data = upgrade_customer_info_eval.loc[:, 'first_activation_date']
upgrade_customer_info_eval['first_activation_year'] = data.apply(lambda x:str(x).split('-')[0])


# In[38]:


#Train
data = upgrade_customer_info.loc[:, 'first_activation_date']
upgrade_customer_info['first_activation_month'] = data.apply(lambda x:str(x).split('-')[1])

#Eval
data = upgrade_customer_info_eval.loc[:, 'first_activation_date']
upgrade_customer_info_eval['first_activation_month'] = data.apply(lambda x:str(x).split('-')[1])


# In[39]:


data = upgrade_customer_info['first_activation_date']
data


# In[40]:


#Train
data = upgrade_customer_info.loc[:, 'first_activation_date']
upgrade_customer_info['first_activation_day'] = data.apply(lambda x:str(x).split('-')[2])

#Eval
data = upgrade_customer_info_eval.loc[:, 'first_activation_date']
upgrade_customer_info_eval['first_activation_day'] = data.apply(lambda x:str(x).split('-')[2])


# In[41]:


#Train
upgrade_customer_info.loc[:,'first_activation_year'].replace('0', np.nan, inplace=True)
upgrade_customer_info.loc[:,'first_activation_month'].replace('0', np.nan, inplace=True)
upgrade_customer_info.loc[:,'first_activation_day'].replace('0', np.nan, inplace=True)

#Eval
upgrade_customer_info_eval.loc[:,'first_activation_year'].replace('0', np.nan, inplace=True)
upgrade_customer_info_eval.loc[:,'first_activation_month'].replace('0', np.nan, inplace=True)
upgrade_customer_info_eval.loc[:,'first_activation_day'].replace('0', np.nan, inplace=True)


# In[42]:


#Train
st=StandardScaler()
data=st.fit_transform(upgrade_customer_info.loc[:,['first_activation_year', 'first_activation_month', 'first_activation_day']])
upgrade_customer_info.loc[:,'first_activation_year'] = data[:, 0]
upgrade_customer_info.loc[:,'first_activation_month'] = data[:, 1]
upgrade_customer_info.loc[:,'first_activation_day'] = data[:, 2]

#Eval
st=StandardScaler()
data=st.fit_transform(upgrade_customer_info_eval.loc[:,['first_activation_year', 'first_activation_month', 'first_activation_day']])
upgrade_customer_info_eval.loc[:,'first_activation_year'] = data[:, 0]
upgrade_customer_info_eval.loc[:,'first_activation_month'] = data[:, 1]
upgrade_customer_info_eval.loc[:,'first_activation_day'] = data[:, 2]


# In[43]:


# 缺失值处理
data = upgrade_customer_info.loc[:,['first_activation_year', 'first_activation_month', 'first_activation_day']]
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)

upgrade_customer_info.loc[:,'first_activation_year'] = data[:, 0]
upgrade_customer_info.loc[:,'first_activation_month'] = data[:, 1]
upgrade_customer_info.loc[:,'first_activation_day'] = data[:, 2]

#Eval
data = upgrade_customer_info_eval.loc[:,['first_activation_year', 'first_activation_month', 'first_activation_day']]
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)

upgrade_customer_info_eval.loc[:,'first_activation_year'] = data[:, 0]
upgrade_customer_info_eval.loc[:,'first_activation_month'] = data[:, 1]
upgrade_customer_info_eval.loc[:,'first_activation_day'] = data[:, 2]


# ## redemption_date

# In[44]:


#Train
upgrade_customer_info.loc[:, 'redemption_date'].fillna("0-0-0", inplace=True)

#Eval
upgrade_customer_info_eval.loc[:, 'redemption_date'].fillna("0-0-0", inplace=True)


# In[45]:


#Train
data = upgrade_customer_info.loc[:, 'redemption_date']
upgrade_customer_info.loc[:,'customer_redemption_year'] = data.apply(lambda x:str(x).split('-')[0])

#Eval
data = upgrade_customer_info_eval.loc[:, 'redemption_date']
upgrade_customer_info_eval.loc[:,'customer_redemption_year'] = data.apply(lambda x:str(x).split('-')[0])


# In[46]:


#Train
data = upgrade_customer_info.loc[:, 'redemption_date']
upgrade_customer_info.loc[:,'customer_redemption_month'] = data.apply(lambda x:str(x).split('-')[1])

#Eval
data = upgrade_customer_info_eval.loc[:, 'redemption_date']
upgrade_customer_info_eval.loc[:,'customer_redemption_month'] = data.apply(lambda x:str(x).split('-')[1])


# In[47]:


#Train
data = upgrade_customer_info.loc[:, 'redemption_date']
upgrade_customer_info.loc[:,'customer_redemption_day'] = data.apply(lambda x:str(x).split('-')[2])

#Eval
data = upgrade_customer_info_eval.loc[:, 'redemption_date']
upgrade_customer_info_eval.loc[:,'customer_redemption_day'] = data.apply(lambda x:str(x).split('-')[2])


# In[48]:


#Train
upgrade_customer_info.loc[:,'customer_redemption_year'].replace('0', np.nan, inplace=True)
upgrade_customer_info.loc[:,'customer_redemption_month'].replace('0', np.nan, inplace=True)
upgrade_customer_info.loc[:,'customer_redemption_day'].replace('0', np.nan, inplace=True)

#Eval
upgrade_customer_info_eval.loc[:,'customer_redemption_year'].replace('0', np.nan, inplace=True)
upgrade_customer_info_eval.loc[:,'customer_redemption_month'].replace('0', np.nan, inplace=True)
upgrade_customer_info_eval.loc[:,'customer_redemption_day'].replace('0', np.nan, inplace=True)


# In[49]:


#Train
st=StandardScaler()
data=st.fit_transform(upgrade_customer_info.loc[:,['customer_redemption_year', 'customer_redemption_month', 'customer_redemption_day']])
upgrade_customer_info.loc[:,'customer_redemption_year'] = data[:, 0]
upgrade_customer_info.loc[:,'customer_redemption_month'] = data[:, 1]
upgrade_customer_info.loc[:,'customer_redemption_day'] = data[:, 2]

#Eval
st=StandardScaler()
data=st.fit_transform(upgrade_customer_info_eval.loc[:,['customer_redemption_year', 'customer_redemption_month', 'customer_redemption_day']])
upgrade_customer_info_eval.loc[:,'customer_redemption_year'] = data[:, 0]
upgrade_customer_info_eval.loc[:,'customer_redemption_month'] = data[:, 1]
upgrade_customer_info_eval.loc[:,'customer_redemption_day'] = data[:, 2]


# In[50]:


# 缺失值处理
data = upgrade_customer_info.loc[:,['customer_redemption_year', 'customer_redemption_month', 'customer_redemption_day']]
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_customer_info.loc[:,'customer_redemption_year'] = data[:, 0]
upgrade_customer_info.loc[:,'customer_redemption_month'] = data[:, 1]
upgrade_customer_info.loc[:,'customer_redemption_day'] = data[:, 2]

#Eval
data = upgrade_customer_info_eval.loc[:,['customer_redemption_year', 'customer_redemption_month', 'customer_redemption_day']]
im = SimpleImputer(missing_values=np.nan, strategy='mean')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_customer_info_eval.loc[:,'customer_redemption_year'] = data[:, 0]
upgrade_customer_info_eval.loc[:,'customer_redemption_month'] = data[:, 1]
upgrade_customer_info_eval.loc[:,'customer_redemption_day'] = data[:, 2]


# ## plan_name

# In[51]:


#Train
upgrade_customer_info['plan_name'].fillna('N', inplace=True)

#Eval
upgrade_customer_info_eval['plan_name'].fillna('N', inplace=True)


# In[52]:


upgrade_customer_info['plan_name'].value_counts()


# In[53]:


#Train

oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_customer_info.loc[:, ['plan_name']])

n_samples = contents.shape[1]
plan_field = []
for i in range(n_samples):
    s = 'plan_' + str(i)
    plan_field.append(s)
    
for i in range(n_samples):
    upgrade_customer_info[plan_field[i]] = contents[:, i]

#Eavlation data set
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_customer_info_eval.loc[:, ['plan_name']])

n_samples = contents.shape[1]
plan_field = []
for i in range(n_samples):
    s = 'plan_' + str(i)
    plan_field.append(s)
    
for i in range(n_samples):
    upgrade_customer_info_eval[plan_field[i]] = contents[:, i]


# ## carrier

# In[54]:


#Train
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_customer_info.loc[:, ['carrier']])

n_samples = contents.shape[1]
carrier_field = []
for i in range(n_samples):
    s = 'carrier_' + str(i)
    carrier_field.append(s)
    
for i in range(n_samples):
    upgrade_customer_info[carrier_field[i]] = contents[:, i]

#Eavlation data set
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_customer_info_eval.loc[:, ['carrier']])

n_samples = contents.shape[1]
carrier_field = []
for i in range(n_samples):
    s = 'carrier_' + str(i)
    carrier_field.append(s)
    
for i in range(n_samples):
    upgrade_customer_info_eval[carrier_field[i]] = contents[:, i]


# In[55]:


upgrade_customer_info = upgrade_customer_info.drop(columns=['upgrade', 'plan_subtype', 'date_observed', 'carrier', 'first_activation_date', 'plan_name', 'redemption_date'])
upgrade_customer_info_eval = upgrade_customer_info_eval.drop(columns=['plan_subtype', 'date_observed', 'carrier', 'first_activation_date', 'plan_name', 'redemption_date'])


# In[56]:


upgrade_customer_info.to_csv(root_folder+"code/data/dev/customer_info_new.csv", header=True,index=None)
upgrade_customer_info_eval.to_csv(root_folder+"code/data/eval/customer_info_new_eval.csv", header=True,index=None)


# # Phone info

# In[57]:


# This path will be active after the launch of the hackathon / day-day-up-unsw / s3://tf-trachack-notebooks/day-day-up-unsw/jupyter/jovyan/
teamname = 'day-day-up-unsw' #'trachack-a-groups-admin-py-tracfone'
data_folder='s3://tf-trachack-data/212/'
# change root_folder to your team's root folder
# s3://tf-trachack-notebooks/<this should be replaced by team name as provided in EMAIL>/jupyter/jovyan/
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'


# In[58]:


upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
phone_info=pd.read_csv(data_folder+"data/dev/phone_info.csv")
customer_info=pd.read_csv(data_folder+"data/dev/customer_info.csv")


# In[59]:


upgrade_phone_info = pd.merge(upgrades, phone_info, on = 'line_id', how = 'inner')
upgrade_customer_info = pd.merge(upgrades, customer_info, on = 'line_id', how = 'inner')


# In[60]:


upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
phone_info_eval=pd.read_csv(data_folder+"data/eval/phone_info.csv")
customer_info_eval=pd.read_csv(data_folder+"data/eval/customer_info.csv")


# In[61]:


upgrade_phone_info_eval = pd.merge(upgrades_eval, phone_info_eval, on = 'line_id', how = 'inner')
upgrade_customer_info_eval = pd.merge(upgrades_eval, customer_info_eval, on = 'line_id', how = 'inner')


# # Data processing
# ## cpu_cores

# In[62]:


#Train
upgrade_phone_info.loc[:, 'cpu_cores'].fillna('0', inplace=True)
upgrade_phone_info.loc[:, 'cpu_cores'] = upgrade_phone_info.loc[:, 'cpu_cores'].apply(lambda x: sum(map(int, x.split('+'))))
upgrade_phone_info.loc[:, 'cpu_cores'].replace(0, np.nan, inplace=True)
# 缺失值处理
upgrade_phone_info.loc[:, 'cpu_cores'].fillna(upgrade_phone_info.loc[:, 'cpu_cores'].mean(), inplace=True)
# st=StandardScaler()
# data=st.fit_transform(upgrade_phone_info_eval.loc[:,['cpu_cores']])
# upgrade_phone_info.loc[:, ['cpu_cores']] = data

#Eval
upgrade_phone_info_eval.loc[:, 'cpu_cores'].fillna('0', inplace=True)
upgrade_phone_info_eval.loc[:, 'cpu_cores'] = upgrade_phone_info_eval.loc[:, 'cpu_cores'].apply(lambda x: sum(map(int, x.split('+'))))
upgrade_phone_info_eval.loc[:, 'cpu_cores'].replace(0, np.nan, inplace=True)
# 缺失值处理
upgrade_phone_info_eval.loc[:, 'cpu_cores'].fillna(upgrade_phone_info_eval.loc[:, 'cpu_cores'].mean(), inplace=True)
# st=StandardScaler()
# data=st.fit_transform(upgrade_phone_info_eval.loc[:,['cpu_cores']])
# upgrade_phone_info_eval.loc[:, ['cpu_cores']] = data


# In[63]:


#Train
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info.loc[:,['cpu_cores']])
upgrade_phone_info.loc[:, ['cpu_cores']] = data

#Eval
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info_eval.loc[:,['cpu_cores']])
upgrade_phone_info_eval.loc[:, ['cpu_cores']] = data


# In[64]:


# 删除多余的列
# upgrade_phone_info.columns
# c1 = cl.delete()
train_set = upgrade_phone_info[['line_id', 'cpu_cores']]
#  = upgrade_phone_info.drop(columns=c1)
# upgrade_phone_info


# In[65]:


# 删除多余的列
eval_set = upgrade_phone_info_eval[['line_id', 'cpu_cores']]
# c1 = cl.delete([0, 3])
# c1
# upgrade_phone_info_eval = upgrade_phone_info.drop(columns=c1)
# upgrade_phone_info_eval


# In[66]:


phone_columns = upgrade_phone_info.columns
phone_columns_eval = upgrade_phone_info_eval.columns


# ## Others features in phone info

# In[67]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[68]:


# le = LabelEncoder()

upgrade_phone_info.loc[:, ['total_ram']] = upgrade_phone_info.loc[:, ['total_ram']].fillna("NEW")
upgrade_phone_info_eval.loc[:, ['total_ram']] = upgrade_phone_info_eval.loc[:, ['total_ram']].fillna("NEW")
upgrade_phone_info.loc[:, ['gsma_model_name']] = upgrade_phone_info.loc[:, ['gsma_model_name']].fillna("NEW")
upgrade_phone_info_eval.loc[:, ['gsma_model_name']] = upgrade_phone_info_eval.loc[:, ['gsma_model_name']].fillna("NEW")
upgrade_phone_info.loc[:, ['manufacturer']] = upgrade_phone_info.loc[:, ['manufacturer']].fillna("NEW")
upgrade_phone_info_eval.loc[:, ['manufacturer']] = upgrade_phone_info_eval.loc[:, ['manufacturer']].fillna("NEW")
upgrade_phone_info.loc[:, ['internal_storage_capacity']] = upgrade_phone_info.loc[:, ['internal_storage_capacity']].fillna("NEW")
upgrade_phone_info_eval.loc[:, ['internal_storage_capacity']] = upgrade_phone_info_eval.loc[:, ['internal_storage_capacity']].fillna("NEW")

upgrade_phone_info.loc[:, ['os_version']] = upgrade_phone_info.loc[:, ['os_version']].fillna("NEW")
upgrade_phone_info_eval.loc[:, ['os_version']] = upgrade_phone_info_eval.loc[:, ['os_version']].fillna("NEW")
# upgrade_phone_info.loc[:, ['os_version']] = upgrade_phone_info.loc[:, ['os_version']].apply(lambda x: le.fit_transform(x))
# upgrade_phone_info_eval.loc[:, ['os_version']] = upgrade_phone_info_eval.loc[:, ['os_version']].apply(lambda x: le.fit_transform(x))

#Train (total_ram)

# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info.loc[:, ['total_ram']])
# pca = PCA(n_components=0.95)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# total_ram_field = []
# for i in range(n_samples):
#     s = 'total_ram_' + str(i)
#     total_ram_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info[total_ram_field[i]] = data[:, i]

# drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['total_ram'])
    
#Eval
# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info_eval.loc[:, ['total_ram']])
# pca = PCA(n_components=0.95)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# total_ram_field = []
# for i in range(n_samples):
#     s = 'total_ram_' + str(i)
#     total_ram_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info_eval[total_ram_field[i]] = data[:, i]

# drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['total_ram'])

#Train (gsma_model_name) （时间太长，不能不用pca，否则分数降低太多） （好像不错，但是我这边结果不行）
# 加了之后明显降低

# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info.loc[:, ['gsma_model_name']])
# pca = PCA(n_components=140)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# total_ram_field = []
# for i in range(n_samples):
#     s = 'gsma_model_name_' + str(i)
#     total_ram_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info[total_ram_field[i]] = data[:, i]
# # drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['gsma_model_name'])
    
# #Eval
# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info_eval.loc[:, ['gsma_model_name']])
# pca = PCA(n_components=140)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# gsma_model_name_field = []
# for i in range(n_samples):
#     s = 'gsma_model_name_' + str(i)
#     gsma_model_name_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info_eval[gsma_model_name_field[i]] = data[:, i]
# drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['gsma_model_name'])


#Train (manufacturer)  (0.926 - 10维度, 不定维度,0.9264)
# 最低0.9219330855018587 最高0.9302832244008715
# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info.loc[:, ['manufacturer']])
# pca = PCA(n_components=10)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# manufacturer_field = []
# for i in range(n_samples):
#     s = 'manufacturer_' + str(i)
#     manufacturer_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info[manufacturer_field[i]] = data[:, i]
# drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['manufacturer'])
    
#Eval
# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info_eval.loc[:, ['manufacturer']])
# pca = PCA(n_components=10)
# data = pca.fit_transform(contents)

# n_samples = data.shape[1]
# manufacturer_field = []
# for i in range(n_samples):
#     s = 'manufacturer_' + str(i)
#     manufacturer_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info_eval[manufacturer_field[i]] = data[:, i]

# drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['manufacturer'])
    
#Train (internal_storage_capacity)  - 最低0.919653893695921， 最高0.9265770423991727

# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info.loc[:, ['internal_storage_capacity']])
# # pca = PCA(n_components=0.95)
# # data = pca.fit_transform(contents)

# n_samples = contents.shape[1]
# internal_storage_capacity_field = []
# for i in range(n_samples):
#     s = 'internal_storage_capacity_' + str(i)
#     internal_storage_capacity_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info[internal_storage_capacity_field[i]] = contents[:, i]
# drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['internal_storage_capacity'])
    
#Eval
# oh = OneHotEncoder(sparse = False)
# contents = oh.fit_transform(upgrade_phone_info_eval.loc[:, ['internal_storage_capacity']])
# # pca = PCA(n_components=0.95)
# # data = pca.fit_transform(contents)

# n_samples = contents.shape[1]
# internal_storage_capacity_field = []
# for i in range(n_samples):
#     s = 'internal_storage_capacity_' + str(i)
#     internal_storage_capacity_field.append(s)
    
# for i in range(n_samples):
#     upgrade_phone_info_eval[internal_storage_capacity_field[i]] = contents[:, i]

# drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['internal_storage_capacity'])
    
    
#Train (os_version) (暂时排除)
# 0.9340540540540541 最高 0.9189189189189189 最低  比较稳定

# upgrade_phone_info.loc[:, ['os_version']] = upgrade_phone_info.loc[:, ['os_version']].fillna("NEW")
# upgrade_phone_info_eval.loc[:, ['os_version']] = upgrade_phone_info_eval.loc[:, ['os_version']].fillna("NEW")


oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_phone_info.loc[:, ['os_version']])
pca = PCA(n_components=0.95)
data = pca.fit_transform(contents)

n_samples = data.shape[1]
os_version_field = []
for i in range(n_samples):
    s = 'os_version_' + str(i)
    os_version_field.append(s)
    
for i in range(n_samples):
    upgrade_phone_info[os_version_field[i]] = data[:, i]
# drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['os_version'])
    
# #Eval
oh = OneHotEncoder(sparse = False)
contents = oh.fit_transform(upgrade_phone_info_eval.loc[:, ['os_version']])
pca = PCA(n_components=0.95)
data = pca.fit_transform(contents)

n_samples = data.shape[1]
os_version_field = []
for i in range(n_samples):
    s = 'os_version_' + str(i)
    os_version_field.append(s)
    
for i in range(n_samples):
    upgrade_phone_info_eval[os_version_field[i]] = data[:, i]

# drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['os_version'])
    
    

### 两个时间都要
    
# Train 缺失值处理 year_released
data = upgrade_phone_info.loc[:, ['year_released']]
im = SimpleImputer(missing_values=np.nan, strategy='median')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_phone_info.loc[:, ['year_released']] = data
# Calculate the time difference
upgrade_phone_info['year_released_1'] = upgrade_phone_info.loc[:, ['year_released']].apply(lambda x:2021-x)
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info.loc[:,['year_released_1']])
upgrade_phone_info['year_released_1'] = data

#Eval
data = upgrade_phone_info_eval.loc[:, ['year_released']]
im = SimpleImputer(missing_values=np.nan, strategy='median')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_phone_info_eval.loc[:, ['year_released']] = data
# Calculate the time difference
upgrade_phone_info_eval['year_released_1'] = upgrade_phone_info_eval.loc[:, ['year_released']].apply(lambda x:2021-x)
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info_eval.loc[:,['year_released_1']])
upgrade_phone_info_eval['year_released_1'] = data

# 光三个特征是0.920
# 0.930 0.921 0.929，0.927
# 0.929 0.925，0.926不算时间差
#Train 缺失值处理 year_released


# Calculate the time difference
upgrade_phone_info.loc[:, ['year_released']] = upgrade_phone_info.loc[:, ['year_released']].apply(lambda x: x)
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info.loc[:,['year_released']])
upgrade_phone_info.loc[:, ['year_released']] = data

#Eval

# Calculate the time difference
upgrade_phone_info_eval.loc[:, ['year_released']] = upgrade_phone_info_eval.loc[:, ['year_released']].apply(lambda x:x)
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info_eval.loc[:,['year_released']])
upgrade_phone_info_eval.loc[:, ['year_released']] = data


#Train 缺失值处理 lte_category
data = upgrade_phone_info.loc[:, ['lte_category']]
im = SimpleImputer(missing_values=np.nan, strategy='median')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_phone_info.loc[:, ['lte_category']] = data
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info.loc[:,['lte_category']])
upgrade_phone_info.loc[:, ['lte_category']] = data

#Eval
data = upgrade_phone_info_eval.loc[:, ['lte_category']]
im = SimpleImputer(missing_values=np.nan, strategy='median')
# im = KNNImputer(n_neighbors=2)
data = im.fit_transform(data)
upgrade_phone_info_eval.loc[:, ['lte_category']] = data
st=StandardScaler()
data=st.fit_transform(upgrade_phone_info_eval.loc[:,['lte_category']])
upgrade_phone_info_eval.loc[:, ['lte_category']] = data


# In[69]:


# Drop columns
upgrade_phone_info = upgrade_phone_info.drop(columns=['date_observed', 'upgrade', 'expandable_storage', 'gsma_device_type', 'gsma_operating_system', 'lte', 'lte_advanced', 'touch_screen', 'wi_fi', 'os_family', 'os_name', 'os_vendor', 'sim_size'])
upgrade_phone_info


# In[70]:


# Drop columns
upgrade_phone_info_eval = upgrade_phone_info_eval.drop(columns=['date_observed', 'expandable_storage', 'gsma_device_type', 'gsma_operating_system', 'lte', 'lte_advanced', 'touch_screen', 'wi_fi', 'os_family', 'os_name', 'os_vendor', 'sim_size'])
upgrade_phone_info_eval


# In[71]:


# To csv
upgrade_phone_info.to_csv(root_folder+"code/data/dev/phone_info.csv", header=True,index=None)
upgrade_phone_info_eval.to_csv(root_folder+"code/data/eval/phone_info_eval.csv", header=True,index=None)
# upgrade_phone_info.to_csv(root_folder+"Hanxi Liu/phone_info.csv", header=True, index=None)
# upgrade_phone_info_eval.to_csv(root_folder+"Hanxi Liu/phone_info_eval.csv", header=True, index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # data preprocessing(lrp_enrollment) Jason

# In[72]:


# merge 3 table of interest in training data
upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv") # table 1
customer_info=pd.read_csv(data_folder+"data/dev/customer_info.csv") # table 2
phone_info=pd.read_csv(data_folder+"data/dev/phone_info.csv") # table 3
upgrades_costomer_info = pd.merge(customer_info, upgrades, on = 'line_id')
upgrades_phone_costomer_info = pd.merge(phone_info, upgrades_costomer_info, on = 'line_id')

# merge 3 table of interest in testing data
upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
customer_info_eval=pd.read_csv(data_folder+"data/eval/customer_info.csv")
phone_info_eval=pd.read_csv(data_folder+"data/eval/phone_info.csv")
upgrades_costomer_info_eval = pd.merge(customer_info_eval, upgrades_eval, on = 'line_id')
upgrades_phone_costomer_info_eval = pd.merge(phone_info_eval, upgrades_costomer_info_eval, on = 'line_id')


# In[73]:


lrp_enrollment = pd.read_csv(data_folder+"data/dev/lrp_enrollment.csv")
lrp_enrollment_eval = pd.read_csv(data_folder+"data/eval/lrp_enrollment.csv")


# In[74]:


def clean_data(data):
    data['lrp_enrollment_year'] = pd.DatetimeIndex(data['lrp_enrollment_date']).year
    data.loc[data['lrp_enrolled'] == 'Y', 'lrp_enrolled'] = 1
    lrp_enrolled_nan = 0
    
    data.loc[data['lrp_enrollment_year'] == 2016, 'lrp_enrollment_year'] = 0
    data.loc[data['lrp_enrollment_year'] == 2017, 'lrp_enrollment_year'] = 0
    data.loc[data['lrp_enrollment_year'] == 2018, 'lrp_enrollment_year'] = 1
    data.loc[data['lrp_enrollment_year'] == 2019, 'lrp_enrollment_year'] = 1
    data.loc[data['lrp_enrollment_year'] == 2020, 'lrp_enrollment_year'] = 2
    data.loc[data['lrp_enrollment_year'] == 2021, 'lrp_enrollment_year'] = 2
    
    lrp_enrollment_year_mean = data['lrp_enrollment_year'].dropna().mean()
    data['lrp_enrolled'] = data['lrp_enrolled'].fillna(lrp_enrolled_nan)
    
    lrp_enrollment_group = data.groupby('line_id').mean()
    print('finished cleaning data')
    return lrp_enrollment_group


# In[75]:


lrp_enrollment_group = clean_data(lrp_enrollment)
lrp_enrollment_group_eval = clean_data(lrp_enrollment_eval)


# In[76]:


upgrades_lrp_enrollment = pd.merge(upgrades, lrp_enrollment_group, on = 'line_id', how='left')
upgrades_lrp_enrollment_eval = pd.merge(upgrades_eval, lrp_enrollment_group_eval, on = 'line_id', how='left')


# In[77]:


def clean_data_merge(data):
    lrp_enrolled_nan = 0
    lrp_enrollment_year_mean = data['lrp_enrollment_year'].dropna().mean()
    data['lrp_enrolled'] = data['lrp_enrolled'].fillna(lrp_enrolled_nan)
    data['lrp_enrollment_year'] = data['lrp_enrollment_year'].fillna(lrp_enrollment_year_mean)
    
    data.loc[data['upgrade'] == 'no', 'upgrade'] = 0
    data.loc[data['upgrade'] == 'yes', 'upgrade'] = 1
    print('finished cleaning data')


# In[78]:


def clean_data_merge_eval(data):
    lrp_enrolled_nan = 0
    lrp_enrollment_year_mean = data['lrp_enrollment_year'].dropna().mean()
    data['lrp_enrolled'] = data['lrp_enrolled'].fillna(lrp_enrolled_nan)
    data['lrp_enrollment_year'] = data['lrp_enrollment_year'].fillna(lrp_enrollment_year_mean)
    
    print('finished cleaning data')


# In[79]:


clean_data_merge(upgrades_lrp_enrollment)
clean_data_merge_eval(upgrades_lrp_enrollment_eval)


# In[80]:


upgrades_lrp_enrollment.to_csv(root_folder+"code/data/dev/lrp_enrollment_new.csv",header=True,index=None)
upgrades_lrp_enrollment_eval.to_csv(root_folder+"code/data/eval/lrp_enrollment_new_eval.csv",header=True,index=None)


# # wuke

# In[84]:


import datetime
lrp_points_info = pd.read_csv(data_folder+"data/dev/lrp_points.csv") # table 4
upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
upgrades_lrp_points_info = pd.merge(upgrades,lrp_points_info, on = 'line_id',how = 'left')

lrp_points_info_eval = pd.read_csv(data_folder+"data/eval/lrp_points.csv") # table 4
upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
upgrades_lrp_points_info_eval = pd.merge(upgrades_eval,lrp_points_info_eval, on = 'line_id',how = 'left')



#数据处理 归一化 quantity, total_quantity , 并处理 status

         
status_na = 0
#取均值
data = upgrades_lrp_points_info
quantity_mean = data['quantity'].dropna().mean()
total_quantity_mean = data['total_quantity'].dropna().mean()

#填补Nan
data['quantity'] = data['quantity'].fillna(quantity_mean)
data['total_quantity'] = data['total_quantity'].fillna(total_quantity_mean)
data['status'] = data['status'].fillna(status_na)


data.loc[data['status'] == 'ENROLLED', 'status'] = 1
#规格化
s = (data['quantity'] - data['quantity'].min())/(data['quantity'].max() - data['quantity'].min())
s1 = (data['total_quantity'] - data['total_quantity'].min())/(data['total_quantity'].max() - data['total_quantity'].min())
new_data = data.drop(['quantity'],axis=1)
new_data = new_data.drop(['total_quantity'],axis=1)
#

new_data.insert(3,'quantity',s)
new_data.insert(5,'total_quantity',s1)


today = datetime.date.today()
new_data['now_day'] = today


new_data['update_date'] = new_data['update_date'].fillna(today)
new_data["days"]=pd.to_datetime(new_data["now_day"])-pd.to_datetime(new_data["update_date"])
new_data["days"]=new_data["days"].map(lambda x:x.days)
final_data = new_data[['line_id','quantity','total_quantity','status','days']].copy()




# In[85]:


# save to CSV
final_data.to_csv(root_folder+"code/data/dev/lrp_points_info.csv",header=True,index=None)


# In[86]:


#eval
#数据处理 归一化 quantity, total_quantity , 并处理 status

         
status_na = 0
#取均值
data_eval = upgrades_lrp_points_info_eval
quantity_mean = data_eval['quantity'].dropna().mean()
total_quantity_mean = data_eval['total_quantity'].dropna().mean()

#填补Nan
data_eval['quantity'] = data_eval['quantity'].fillna(quantity_mean)
data_eval['total_quantity'] = data_eval['total_quantity'].fillna(total_quantity_mean)
data_eval['status'] = data_eval['status'].fillna(status_na)


data_eval.loc[data_eval['status'] == 'ENROLLED', 'status'] = 1
#规格化
s = (data_eval['quantity'] - data_eval['quantity'].min())/(data_eval['quantity'].max() - data_eval['quantity'].min())
s1 = (data_eval['total_quantity'] - data_eval['total_quantity'].min())/(data_eval['total_quantity'].max() - data_eval['total_quantity'].min())
new_data_eval = data_eval.drop(['quantity'],axis=1)
new_data_eval = new_data_eval.drop(['total_quantity'],axis=1)
#
new_data_eval.insert(3,'quantity',s)
new_data_eval.insert(5,'total_quantity',s1)





today = datetime.date.today()
new_data_eval['now_day'] = today
new_data_eval['update_date'] = new_data_eval['update_date'].fillna(today)
new_data_eval["days"]=pd.to_datetime(new_data_eval["now_day"])-pd.to_datetime(new_data_eval["update_date"])
new_data_eval["days"]=new_data_eval["days"].map(lambda x:x.days)


final_data_eval = new_data_eval[['line_id','quantity','total_quantity','status','days']].copy()


# In[87]:


#save to CSV



# # Zhengyu Luo

# In[88]:


upgrades=pd.read_csv(data_folder+"data/dev/upgrades.csv")
customer_info=pd.read_csv(data_folder+"data/dev/customer_info.csv")
network_usage_domestic=pd.read_csv(data_folder+"data/dev/network_usage_domestic.csv")
suspensions=pd.read_csv(data_folder+"data/dev/suspensions.csv",parse_dates=['suspension_start_date','suspension_end_date'])
redemptions=pd.read_csv(data_folder+"data/dev/redemptions.csv",parse_dates=['redemption_date'])

customer_info = customer_info[['line_id','carrier','plan_name']].copy()


# In[89]:


upgrades_eval=pd.read_csv(data_folder+"data/eval/upgrades.csv")
customer_info_eval=pd.read_csv(data_folder+"data/eval/customer_info.csv")
suspensions_eval=pd.read_csv(data_folder+"data/eval/suspensions.csv",parse_dates=['suspension_start_date','suspension_end_date'])
network_usage_domestic_eval=pd.read_csv(data_folder+"data/eval/network_usage_domestic.csv")
redemptions_eval=pd.read_csv(data_folder+"data/eval/redemptions.csv",parse_dates=['redemption_date'])


customer_info_eval = customer_info_eval[['line_id','carrier','plan_name']].copy()


# In[90]:


def clean_data(data):
    data.loc[data['carrier'] == 'carrier 1', 'carrier'] = 0
    data.loc[data['carrier'] == 'carrier 2', 'carrier'] = 1
    data.loc[data['carrier'] == 'carrier 3', 'carrier'] = 1
      
    data.loc[data['plan_name'] == 'plan 1', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 2', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 3', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 4', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'Other', 'plan_name'] = 0
    plan_name_nan = 1 
   
    voice_min_total_median = data['voice_min_total'].dropna().median()
    voice_count_total_median = data['voice_count_total'].dropna().median()
    total_kb_median = data['total_kb'].dropna().median()
    messages_median = data['messages'].dropna().median()

    
    data['voice_min_total'] = data['voice_min_total'].fillna(0)
    data['voice_count_total'] = data['voice_count_total'].fillna(voice_count_total_median)
    data['total_kb'] = data['total_kb'].fillna(total_kb_median)
    data['messages'] = data['messages'].fillna(messages_median)
    data['plan_name'] = data['plan_name'].fillna(plan_name_nan)
    
    
    # target
    data.loc[data['upgrade'] == 'no', 'upgrade'] = 0
    data.loc[data['upgrade'] == 'yes', 'upgrade'] = 1
    print('finished training data')


# In[91]:


network_usage_domestic['voice_min_total'] = network_usage_domestic['voice_min_in'] + network_usage_domestic['voice_min_out']
network_usage_domestic['messages'] = network_usage_domestic['sms_in']

network_usage_domestic = network_usage_domestic.groupby(['line_id'])['messages',"voice_min_total","voice_count_total","total_kb"].mean()
network_usage_domestic = pd.merge(upgrades,network_usage_domestic, on = 'line_id', how = 'left')
network_usage_domestic = pd.merge(network_usage_domestic,customer_info, on = 'line_id', how = 'left')
network_usage_domestic = network_usage_domestic[['line_id','upgrade','messages','voice_min_total','voice_count_total','total_kb','carrier','plan_name']].copy()


# In[92]:


clean_data(network_usage_domestic)


# In[93]:


def clean_data_eval(data):
    data.loc[data['carrier'] == 'carrier 1', 'carrier'] = 0
    data.loc[data['carrier'] == 'carrier 2', 'carrier'] = 1
    data.loc[data['carrier'] == 'carrier 3', 'carrier'] = 1
      
    data.loc[data['plan_name'] == 'plan 1', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 2', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 3', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'plan 4', 'plan_name'] = 0
    data.loc[data['plan_name'] == 'Other', 'plan_name'] = 0
    plan_name_nan = 1
    
    voice_count_total_median = data['voice_count_total'].dropna().median()
    total_kb_median = data['total_kb'].dropna().median()
    messages_median = data['messages'].dropna().median()

    
    data['voice_min_total'] = data['voice_min_total'].fillna(0)
    data['voice_count_total'] = data['voice_count_total'].fillna(voice_count_total_median)
    data['total_kb'] = data['total_kb'].fillna(total_kb_median)
    data['messages'] = data['messages'].fillna(messages_median)
    data['plan_name'] = data['plan_name'].fillna(plan_name_nan)
    
   
    data['plan_name'] = data['plan_name'].fillna(plan_name_nan)

  
    print('finished training data')


# In[94]:


network_usage_domestic_eval['voice_min_total'] = network_usage_domestic_eval['voice_min_in'] + network_usage_domestic_eval['voice_min_out']
network_usage_domestic_eval['messages'] = network_usage_domestic_eval['sms_in']

network_usage_domestic_eval = network_usage_domestic_eval.groupby(['line_id'])['messages',"voice_min_total","voice_count_total","total_kb"].mean()
network_usage_domestic_eval = pd.merge(upgrades_eval,network_usage_domestic_eval, on = 'line_id', how = 'left')
network_usage_domestic_eval = pd.merge(network_usage_domestic_eval,customer_info_eval, on = 'line_id', how = 'left')
network_usage_domestic_eval = network_usage_domestic_eval[['line_id','messages','voice_min_total','voice_count_total','total_kb','carrier','plan_name']].copy()


# In[95]:


clean_data_eval(network_usage_domestic_eval)


# In[96]:


sus = suspensions.copy()
sus['time'] = (sus['suspension_end_date'] - sus['suspension_start_date']).dt.days
sus = sus.groupby(['line_id'])['time'].sum()
sus = pd.merge(sus,upgrades,on='line_id',how='left')
sus = sus[['line_id','time']].copy()
network_usage_domestic = pd.merge(network_usage_domestic,sus,on='line_id',how='left')

time_median = network_usage_domestic['time'].dropna().median()
network_usage_domestic['time'] = network_usage_domestic['time'].fillna(0)


# In[97]:


sus_eval = suspensions_eval.copy()
sus_eval['time'] = (sus_eval['suspension_end_date'] - sus_eval['suspension_start_date']).dt.days
sus_eval = sus_eval.groupby(['line_id'])['time'].sum()
network_usage_domestic_eval = pd.merge(network_usage_domestic_eval,sus_eval,on='line_id',how='left')

time_median = network_usage_domestic_eval['time'].dropna().median()
network_usage_domestic_eval['time'] = network_usage_domestic_eval['time'].fillna(time_median)


# In[98]:


redp = redemptions[['line_id','redemption_date']].copy()
redemption_date = redp.groupby(['line_id'])['redemption_date'].max()
redemption_times = redp.groupby(['line_id'])['redemption_date'].count()


# In[99]:


redp= pd.merge(redemption_date,upgrades,on='line_id',how='left')
redp.rename(columns={'redemption_date':'last_redemption_date'}, inplace = True)
redp=pd.merge(redp,redemption_times,on='line_id',how='left')
redp.rename(columns={'redemption_date':'redemption_times'}, inplace = True)


# In[100]:


redp['last_redemption'] = (pd.to_datetime('today') - redp['last_redemption_date']).apply(lambda x:x.days)
redp = redp[['line_id','last_redemption','redemption_times']].copy()


# In[101]:


network_usage_domestic = pd.merge(network_usage_domestic,redp,on='line_id',how='left')


# In[102]:


redemption_times_median = network_usage_domestic['redemption_times'].dropna().median()
network_usage_domestic['redemption_times'] = network_usage_domestic['redemption_times'].fillna(0)
last_redemption_median = network_usage_domestic['last_redemption'].dropna().median()
network_usage_domestic['last_redemption'] = network_usage_domestic['last_redemption'].fillna(last_redemption_median)


# In[103]:


redp_eval = redemptions_eval[['line_id','redemption_date']].copy()
redemption_date_eval = redp_eval.groupby(['line_id'])['redemption_date'].max()
redemption_times_eval = redp_eval.groupby(['line_id'])['redemption_date'].count()


# In[104]:


redp_eval= pd.merge(redemption_date_eval,upgrades_eval,on='line_id',how='left')
redp_eval.rename(columns={'redemption_date':'last_redemption_date'}, inplace = True)
redp_eval=pd.merge(redp_eval,redemption_times_eval,on='line_id',how='left')
redp_eval.rename(columns={'redemption_date':'redemption_times'}, inplace = True)


# In[105]:


redp_eval['last_redemption'] = (pd.to_datetime('today') - redp_eval['last_redemption_date']).apply(lambda x:x.days)
redp_eval = redp_eval[['line_id','last_redemption','redemption_times']].copy()


# In[106]:


network_usage_domestic_eval = pd.merge(network_usage_domestic_eval,redp_eval,on='line_id',how='left')


# In[107]:


redemption_times_median = network_usage_domestic_eval['redemption_times'].dropna().median()
network_usage_domestic_eval['redemption_times'] = network_usage_domestic_eval['redemption_times'].fillna(redemption_times_median)
last_redemption_median = network_usage_domestic_eval['last_redemption'].dropna().median()
network_usage_domestic_eval['last_redemption'] = network_usage_domestic_eval['last_redemption'].fillna(last_redemption_median)


# In[108]:


network_usage_domestic.to_csv(root_folder+"code/data/dev/network_usage_domestic_new.csv",header=True,index=None)
network_usage_domestic_eval.to_csv(root_folder+"code/data/eval/network_usage_domestic_new_eval.csv",header=True,index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Table merge

# In[130]:


##### "Dev" preparation
# No.1: the code is in the route: code/data_cleaning
upgrades_customer_info_new = pd.read_csv(root_folder + "code/data/dev/customer_info_new.csv" ).copy()
# No.2: the code is in the route: code/data_cleaning
upgrades_hanxi = pd.read_csv(root_folder+"code/data/dev/upgrade_redemptions_deactivations_reactivations_suspensions.csv").copy()
# NO.3: the code is in the route: code/data_cleaning/luke_data_model_5
upgrades_luke = pd.read_csv(root_folder + "code/data/dev/luke_data_model_5.csv").copy()
# NO.4: the code is in the route: code/data_cleaning/lrp_enrollment
upgrades_jason = pd.read_csv(root_folder + "code/data/dev/lrp_enrollment_new.csv").copy()
# NO.5: the code is in the route: code/data_cleaning/lrp_points
upgrades_wuke = pd.read_csv(root_folder + "code/data/dev/lrp_points_info.csv").copy()
# No.6: the code is in the route: code/data_cleaning
upgrades_ace = pd.read_csv(root_folder + "code/data/dev/phone_info.csv").copy()


# In[132]:


##### "Eval" preparation
# No.1: the code is in the route: code/data_cleaning
upgrades_customer_info_new_eval = pd.read_csv(root_folder + "code/data/eval/customer_info_new_eval.csv" ).copy()
# No.2: the code is in the route: code/data_cleaning
upgrades_hanxi_eval = pd.read_csv(root_folder+"code/data/eval/upgrade_redemptions_deactivations_reactivations_suspensions_eval.csv").copy()
# NO.3: the code is in the route: code/data_cleaning/
upgrades_luke_eval = pd.read_csv(root_folder + "code/data/eval/luke_data_model_eval_5.csv").copy()
# NO.4: the code is in the route: code/data_cleaning/
upgrades_jason_eval = pd.read_csv(root_folder + "code/data/eval/lrp_enrollment_new_eval.csv").copy()
# NO.5: the code is in the route: code/data_cleaning/
upgrades_wuke_eval = pd.read_csv(root_folder + "code/data/eval/lrp_points_info_eval.csv").copy()
# No.6: the code is in the route: code/data_cleaning
upgrades_ace_eval = pd.read_csv(root_folder + "code/data/eval/phone_info_eval.csv").copy()


# # model training

# # 1.1 "Dev"

# In[133]:


#####  'a' : "network_usage + redemptions..."

a = upgrades_hanxi.merge(upgrades_luke.drop(columns = ['carrier', 'plan_name']),on = 'line_id')
a = a.drop(columns = ['upgrade_y'])
a = a.rename(columns = {'upgrade_x':'upgrade'})
a.shape


# In[134]:


#####  'b' : "a + phone_info"
b = a.merge(upgrades_ace,on = 'line_id')
print('the number of phone_info : ',b.shape)


# In[135]:


#####  'c' : "b + lrp_part_1"
c = b.merge(upgrades_jason[['line_id','lrp_enrolled','lrp_enrollment_year']], on = 'line_id')

#####  'd' : "c + lrp_part_2"
d = c.merge(upgrades_wuke, on = 'line_id')
d.shape


# In[136]:


##### f: "d + customer"
f = d.merge(upgrades_customer_info_new, on = 'line_id')
f.shape


# In[115]:


f.columns


# # 1.2 "Eval"

# In[137]:


#####  'a' : "network_usage_eval + redemptions..."

a_eval = upgrades_hanxi_eval.merge(upgrades_luke_eval.drop(columns = ['carrier', 'plan_name']),on = 'line_id')

a_eval.shape


# In[138]:


#####  'b' : "a + phone_info"
b_eval = a_eval.merge(upgrades_ace_eval,on = 'line_id')
print('the number of phone_info : ',b_eval.shape)


# In[139]:


#####  'c' : "b + lrp_part_1"
c_eval = b_eval.merge(upgrades_jason_eval[['line_id','lrp_enrolled','lrp_enrollment_year']], on = 'line_id')

#####  'd' : "c + lrp_part_2"
d_eval = c_eval.merge(upgrades_wuke_eval, on = 'line_id')
d_eval.shape


# In[140]:


#####  'f' : "d + customer_info_eval"
f_eval = d_eval.merge(upgrades_customer_info_new_eval, on = 'line_id') # New model
f_eval.shape


# In[120]:


f_eval.columns


# # 2 mature_model "a_b_c_d_f"

# In[141]:



# Confirmed 
f = f.drop(columns = ['status','reactivation_day','deactivation_day','suspension_end_day',
                       'redemption_day','suspension_start_day','first_activation_day',
                       'customer_redemption_day'])
# Try
#f2 = f2.drop(columns = ['suspension_start_day'])


df_selected_ace = f.iloc[:, 1:]

final_x_ace = df_selected_ace.drop(columns=['upgrade'])
final_y_ace = df_selected_ace.loc[:, ['upgrade']].values.astype(np.int32).ravel()


# In[122]:


final_x_ace.columns


# In[123]:


##### Input size #####
final_x_ace.shape


# In[142]:


print('#####Begain#####\n')
for i in range(28,29): # 2 times of split
    print('test:', i)
    set_test_size = i / 1000
    x_train, x_test, y_train, y_test = train_test_split(final_x_ace, final_y_ace, test_size = set_test_size, random_state = 23) # do not shuffle first, ,shuffle = True

    classifier_random_forest = RandomForestClassifier(max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        n_estimators=600, 
        random_state= 11 )
    classifier_random_forest.fit(x_train, y_train)
                         ###########
    random_forest_pred = classifier_random_forest.predict(x_test)
                         ###########
    random_forest_precision = metrics.precision_score(y_test, random_forest_pred, labels = None, pos_label=1, average="binary", sample_weight=None, zero_division =1)
    random_forest_recall = metrics.recall_score(y_test, random_forest_pred)
    a = confusion_matrix(y_test,random_forest_pred)
    print('test_size: {0}, random_forest #####f1_score#####: {1}'.format((i/100), f1_score(y_test,random_forest_pred)))
    print()
    print('random_forest #precision#: ',random_forest_precision)
    print('random_forest #recall#: ',random_forest_recall)
    print('matrix\n', a)
    print()
print('\n#####End#####')


# # 3 Submission

# In[143]:


### Data Preparation
submission = f_eval.copy()

# Confirmed 
submission = submission.drop(columns = ['status','reactivation_day','suspension_end_day','deactivation_day',
                                       'redemption_day','suspension_start_day','first_activation_day',
                                       'customer_redemption_day'])


##### Prediction_1

predicted_data_eval = submission.iloc[:, 1:]   # Same to "df_selected_3" / "final_x3"
predicted_data_eval.shape


# In[144]:


##### Prediction_2
submission['upgrade'] = classifier_random_forest.predict(predicted_data_eval)    # New Column "upgrade"



# In[127]:


to_csv_data = submission[['line_id','upgrade']]
to_csv_data


# In[145]:


##### Distribution
predicted_distribution = submission.upgrade.value_counts(normalize = True)
predicted_distribution = pd.DataFrame(predicted_distribution)
predicted_distribution


# In[ ]:


# This line for submission
#to_csv_data.to_csv(root_folder+"submission/2021-04-25.csv",header=True,index=None)
#submission_path  = root_folder+"submission/2021-04-25.csv"
#print(f"submission saved to {submission_path}")

