import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split


def get_first_dig(num):
    r = num.split('.')
    print(r)
    return int(r[0])


data = pd.read_pickle("data_vec1d.pkl")

print(data)
x = data.iloc[:, 2:]

y = data['ec_number'].apply(get_first_dig).to_numpy()

for elem in y:
    print(elem)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = svm.SVC()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)
