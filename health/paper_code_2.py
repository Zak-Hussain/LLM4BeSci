from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, dat['labels'], test_size=.2, random_state=42
)

# Initializing ridge regression, fit, and evaluate
regr = RidgeCV()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

