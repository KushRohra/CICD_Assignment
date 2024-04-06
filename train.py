import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/train.csv")

X = df.drop(columns=['Disease']).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# model = LogisticRegression(random_state=42, penalty='l2', C=1.0).fit(X_scaled, y)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(random_state=42, penalty='l2'), param_grid, cv=5)
grid_search.fit(X_scaled, y)
best_model = grid_search.best_estimator_

with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)