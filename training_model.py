import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict=pickle.load(open('./data.pickle','rb'))

data = []
labels = []

for sample, label in zip(data_dict['data'], data_dict['labels']):
    if isinstance(sample, (list, np.ndarray)):
        if len(sample) == 42:  # one hand
            sample = sample + [0] * 42  # pad second hand
        elif len(sample) != 84:
            continue  # skip invalid samples
        data.append(sample)
        labels.append(label)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42,shuffle=True,stratify=labels)

model=RandomForestClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(f"{score*100}%")

f=open('model.p','wb')
pickle.dump({'model':model},f)
f.close()


# print(data_dict.keys())
# print(data_dict)