from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import joblib
import os

import nltk
#nltk.download('punkt')

# Load data
data_file_name = 'Product Review Large Data.csv'
data = pd.read_csv(data_file_name)
data = data.dropna(subset=['review_text', 'review_rating'])

# Define target and features
data['sentiment'] = data['review_rating'].apply(lambda x: 'positive' if x > 3 else 'negative')


# Define the path for vectorizer, and classifier files based on the dataset file name
dataset_file_name = os.path.splitext(os.path.basename(data_file_name))[0]
vectorizer_path = f"{dataset_file_name}_vectorizer.pkl"
classifier_path = f"{dataset_file_name}_stack_classifier.pkl"

# Preprocess text data
data['review_text'] = data['review_text'].apply(lambda x: x.lower())
data['review_text'] = data['review_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
ps = PorterStemmer()
data['review_text'] = data['review_text'].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)]))
stopwords = set(ENGLISH_STOP_WORDS)
data['review_text'] = data['review_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))



# Split the data into target and features
X = data['review_text']
y = data['sentiment']

# Balance classes using Random Over-Sampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X.values.reshape(-1, 1), y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled.flatten(), y_resampled, test_size=0.2, random_state=42)

# Check if the vectorizer file exists
if os.path.exists(vectorizer_path):
    # Load the vectorizer from the file
    vectorizer = joblib.load(vectorizer_path)
else:
    # If the vectorizer file does not exist, fit the vectorizer and save it
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)  
    vectorizer.fit(X_train)
    # Save the vectorizer to the file
    joblib.dump(vectorizer, vectorizer_path)

# Check if the classifier file exists
if os.path.exists(classifier_path):
    # Load the classifier from the file
    stacking_classifier = joblib.load(classifier_path)
else:
    # Define base classifiers
    nb_classifier = MultinomialNB()
    rf_classifier = RandomForestClassifier(n_estimators=3,random_state=42)
    svm_classifier = SVC(kernel='linear', probability=True)
    dt_classifier = DecisionTreeClassifier(random_state=42)
    knn_classifier = KNeighborsClassifier()
    lr_classifier = LogisticRegression(max_iter=1000)
    
    # If the classifier file does not exist, train and save the classifier
    base_classifiers = [
        ('nb', nb_classifier),
        ('rf', rf_classifier),
        ('svm', svm_classifier),
        ('dt', dt_classifier),
        ('knn', knn_classifier),
        ('lr', lr_classifier)
    ]
    #rf as meta final estimator
    stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=rf_classifier, stack_method='predict_proba') 
    stacking_classifier.fit(vectorizer.transform(X_train), y_train)
    # Save the classifier to the file
    joblib.dump(stacking_classifier, classifier_path)

# Get indices of non-NaN values in X_test
non_nan_indices = ~pd.isnull(X_test)

# Make predictions only if there are non-NaN values in X_test
if non_nan_indices.any():
    # Make predictions
    stacking_predictions = stacking_classifier.predict(vectorizer.transform(X_test[non_nan_indices]))

    # Evaluate the model
    accuracy_stacking = accuracy_score(y_test[non_nan_indices], stacking_predictions)
    precision_stacking = precision_score(y_test[non_nan_indices], stacking_predictions, average='weighted')
    recall_stacking = recall_score(y_test[non_nan_indices], stacking_predictions, average='weighted')
    f1_stacking = f1_score(y_test[non_nan_indices], stacking_predictions, average='weighted')

    # Print evaluation metrics
    print("Stacking Classifier - Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(
        accuracy_stacking, precision_stacking, recall_stacking, f1_stacking))
else:
    print("No non-NaN values found in X_test.")