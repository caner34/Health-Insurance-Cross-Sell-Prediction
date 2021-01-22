
import pandas as pd
import numpy as np
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt




#############################
#############################
####                     ####
####      TODO LIST      ####
####  similarity matrix  ####
#### classifier pipeline ####
####     up-sampling     ####
####                     ####
#############################
#############################


###########################################
###########################################
####                                   ####
####     EXPLORATORY DATA ANALYSIS     ####
####                                   ####
###########################################
###########################################

# source:
# https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

data = pd.read_csv(os.path.join('data', 'train.csv'))
data.name = 'data'
data = data.drop(columns=['id'])



# SHAPE
# (381109, 12)
data.shape

# Data Types
data.dtypes

# Column Names:
data.columns


stats_descriptive = data.describe()

stats_gender_distribution = pd.DataFrame(data.Gender.value_counts())
stats_response_distribution = pd.DataFrame(data.Response.value_counts())
stats_Vehicle_Damage_distribution = pd.DataFrame(data.Vehicle_Damage.value_counts())
stats_Driving_License_distribution = pd.DataFrame(data.Driving_License.value_counts())
stats_Vintage_distribution = pd.DataFrame(data.Vintage.value_counts())

# they have limited number of unique values among almost 400k samples, which strongly indicates that toose are categorical features
stats_no_unique_regions = len(data.Region_Code.unique())
stats_no_unique_policy_sales_channel = len(data.Policy_Sales_Channel.unique())
stats_no_unique_vintage = len(data.Vintage.unique())


data.head()




###########################################
###########################################
####                                   ####
####           PLOTS & GRAPHS          ####
####                                   ####
###########################################
###########################################


from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
def plot_correlation_matrix(df, size=16):
    df = df.copy()
    encoder = LabelEncoder()
    feature_columns_to_be_made_numerical = [c for c in df.columns if is_string_dtype(df[c])]
    for c in feature_columns_to_be_made_numerical:
        df[c] = encoder.fit_transform(df[c])
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(size,size))  
    sns.heatmap(corrMatrix, annot=True, ax=ax)
    plt.show()

def donut_plot(data, title):
    names = list(data.index)
    size = data[data.columns[0]].tolist()
    # Create a circle for the center of the plot
    my_circle=plt.Circle( (0,0), 0.6, color='white')
    # Give color names
    plt.pie(size, labels=names, colors=['green','red','blue','skyblue'][:len(names)])
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(title.replace('_',' '))
    plt.show()

from pandas.plotting import table
def plot_data_frame(df, title):
    # set fig size
    fig, ax = plt.subplots(figsize=(40, 4.5)) 
    ax.set_title(title,pad=40, fontdict={'fontsize':48})
    # no axes
    ax.xaxis.set_visible(False)  
    ax.yaxis.set_visible(False)  
    # no frame
    ax.set_frame_on(False)  
    # plot table
    tab = table(ax, df, loc='upper right')  
    # set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(20)
    tab.scale(1, 4)


plot_correlation_matrix(data)
stats_response_distribution.plot.pie(y='Response', title='Target Value Distribution')
stats_gender_distribution.plot.pie(y='Gender', title='Gender')
donut_plot(stats_Vehicle_Damage_distribution, 'Vechile_Damage')
plot_data_frame(stats_descriptive, 'Descriptive Statistics')
data.Vintage.plot.kde(bw_method=0.15)
sns.distplot(data.Age, hist=False, hist_kws={'range': (18, 85)})
# normal boxplot
data.Annual_Premium.plot.box()
# boxplot with rarified data
data[data.index % 100 == 0].Annual_Premium.plot.box()







