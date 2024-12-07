import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Example initial data to fit the preprocessor
initial_data = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
    'col3': [7, 8, 9],
    'Age': ['25', '30', '35'],
    'Gender': ['M', 'F', 'M'],
    'Ethnicity': ['Group1', 'Group2', 'Group1'],
    'EducationLevel': ['Bachelors', 'Masters', 'PhD']
})

numeric_col = ['col1', 'col2', 'col3']

num = Pipeline([
    ('scl', StandardScaler())
])

cat = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num, numeric_col),
    ('cat', cat, ['Age', 'Gender', 'Ethnicity', 'EducationLevel'])
])

# Fit and transform initial data
X_processed = preprocessor.fit_transform(initial_data)
y = [0, 1, 0]  # Example target variable (replace with actual target)

# Split the processed data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)
