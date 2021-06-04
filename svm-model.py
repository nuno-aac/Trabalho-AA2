import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split


def get_first_dig(num):
    r = num.split('.')
    return int(r[0])


data = pd.read_pickle("parsed_data/data1-2.pkl")

print(data)
x = data.iloc[:, 1:]

y = data['ec_number'].apply(get_first_dig).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = svm.SVC(C=3)

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
