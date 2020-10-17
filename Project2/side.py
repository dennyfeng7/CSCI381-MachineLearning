import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def side():
    load_breast_cancer
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    # train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # test the model
    predictions = model.predict(x_test)
    print(predictions)

    print(y_test)

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    data = sns.load_dataset("iris")
    data.head()
    # prep the training set
    # X = feature values , all of the columsn except the last column
    X = data.iloc[:, :-1]

    # Y = target values

    y = data.iloc[:, -1]
    plt.xlabel('Features')
    plt.ylabel('Species')

    pltX = data.loc[:, 'sepal_length']
    pltY = data.loc[:, 'species']
    plt.scatter(pltX, pltY, color='blue', Label='sepal_length')

    pltX = data.loc[:, 'sepal_width']
    pltY = data.loc[:, 'species']
    plt.scatter(pltX, pltY, color='green', Label='sepal_width')

    pltX = data.loc[:, 'petal_length']
    pltY = data.loc[:, 'species']
    plt.scatter(pltX, pltY, color='yellow', Label='petal_length')

    pltX = data.loc[:, 'petal_width']
    pltY = data.loc[:, 'species']
    plt.scatter(pltX, pltY, color='orange', Label='petal_width')

    plt.legend(loc=4, prop={'size': 8})
    plt.show()
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    # train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # test the model
    predictions = model.predict(x_test)
    print(predictions)

    print(y_test)


if __name__ == '__main__':
    side()
