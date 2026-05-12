from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# load data
X, y = []
model = None

# parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# set up GridSearchCV
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation
    scoring='accuracy'
)

# run search
grid.fit(X, y)

# access results
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
