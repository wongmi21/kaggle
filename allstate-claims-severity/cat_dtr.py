import time
import pandas
import numpy as np
import csv
from sklearn.model_selection import train_test_split

start_time = time.time()

# Load data
df_data = pandas.read_csv("data/data.csv", header=0)
y = df_data.pop("loss").tolist()

for col in df_data.columns:
    if not col.startswith("cat"):
        df_data.pop(col)

X = df_data.as_matrix().tolist()

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit linear regression model
lr = DecisionTreeRegressor().fit(X_train, y_train)

# Predict test
y_predict_test = lr.predict(X_test)
print(np.mean(abs(y_predict_test - y_test)))

# Predict validation
df_predict = pandas.read_csv("data/predict.csv", header=0)
predict_id = df_predict.pop("id").tolist()
for col in df_predict.columns:
    if not col.startswith("cat"):
        df_predict.pop(col)
X_predict = df_predict.as_matrix().tolist()

y_predict_validation = lr.predict(X_predict)

# Write csv
rows = zip(predict_id, y_predict_validation)

with(open("data/submission_cont_dtr.csv", "wt", encoding="utf8", newline='')) as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['id', 'loss'])
    for row in rows:
        csvwriter.writerow(row)

print("--- %s seconds ---" % (time.time() - start_time))
