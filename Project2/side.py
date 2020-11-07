# Import our libraries

# Pandas and numpy for data wrangling
import inline as inline
import matplotlib
import pandas as pd
import numpy as np

# Seaborn / matplotlib for visualization
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
# %matplotlib inline


# Helper function to split our data
from sklearn.model_selection import train_test_split

# This is our Logit model
from sklearn.linear_model import LogisticRegression

# Helper fuctions to evaluate our model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

df=pd.read_csv('data.csv')

df.head()