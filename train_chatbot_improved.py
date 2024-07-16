import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import ssl

# Disable SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

print("Loading data...")

# Load the data from the CSV file
df = pd.read_csv('chatbot_data.csv')

print("Handling missing values...")

# Handle missing values by dropping rows with any missing values
df.dropna(inplace=True)

# Define a function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

print("Preprocessing text...")

# Apply the preprocessing function to the 'User Input' column
df['User Input'] = df['User Input'].apply(preprocess_text)

print("Text preprocessing completed.")

print("Splitting data into training and test sets...")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['User Input'], df['Response'], test_size=0.2, random_state=42)

# Define the pipeline with TfidfVectorizer and SVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC())
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 2],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

print("Starting GridSearchCV...")

# Use GridSearchCV for hyperparameter tuning with all CPU cores
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
grid_search.fit(X_train, y_train)

print("GridSearchCV completed.")

# Print the best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

print("Evaluating the model on the test set...")

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

print("Saving the best model...")

# Save the best model
with open('chatbot_model_improved.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)

print("Improved model trained and saved successfully.")
