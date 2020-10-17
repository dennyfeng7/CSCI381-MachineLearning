# Import dependencies
import matplotlib.pyplot as plt
import seaborn as sns # Where we get Iris Dataset from
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data set
data = sns.load_dataset("iris")
data.head()
