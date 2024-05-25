from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import nltk

# Assuming nltk has been previously downloaded
nltk.download('punkt')

app = Flask(__name__)

# Define mapping of model names to display names
model_names_mapping = {
    'nb': 'Naive Bayes',
    'svm': 'SVM',
    'dt': 'Decision Tree',
    'rf': 'Random Forest',
    'knn': 'KNN',
    'lr': 'Logistic Regression',
    'stacking': 'Hybrid Model'
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in word_tokenize(text)])
    stopwords = set(ENGLISH_STOP_WORDS)
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['review_text', 'review_rating'])

    # Preprocess text data
    data['review_text'] = data['review_text'].apply(preprocess_text)

    # Define target and features
    data['sentiment'] = data['review_rating'].apply(lambda x: 1 if x > 3 else 0)
    X = data['review_text']
    y = data['sentiment']

    # Balance classes using Random Over-Sampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X.values.reshape(-1, 1), y)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled.flatten(), y_resampled, test_size=0.2, random_state=42)

    # Vectorize text data using TF-IDF
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    vectorizer_path = f"{dataset_name}_vectorizer.pkl"
    
    if os.path.exists(vectorizer_path):
        # Load the vectorizer from the file
        vectorizer = joblib.load(vectorizer_path)
    else:
        # If the vectorizer file does not exist, fit the vectorizer and save it
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)  
        vectorizer.fit(X_train)
        # Save the vectorizer to the file
        joblib.dump(vectorizer, vectorizer_path)

    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, file_path):
    # Define base classifiers
    nb_classifier = MultinomialNB()
    rf_classifier = RandomForestClassifier(n_estimators=3, random_state=42)
    svm_classifier = SVC(kernel='linear', probability=True)
    dt_classifier = DecisionTreeClassifier(random_state=42)
    knn_classifier = KNeighborsClassifier()
    lr_classifier = LogisticRegression(max_iter=1000)
    
    base_classifiers = [
        ('nb', nb_classifier),
        ('rf', rf_classifier),
        ('svm', svm_classifier),
        ('dt', dt_classifier),
        ('knn', knn_classifier),
        ('lr', lr_classifier)
    ]
    
    # Define the stacking classifier with RandomForest as the final estimator
    stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=RandomForestClassifier(random_state=42), stack_method='predict_proba')
    
    # Check if classifier file exists
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    classifier_path = f"{dataset_name}_stack_classifier.pkl"
    
    if os.path.exists(classifier_path):
        # Load the trained classifier if it exists
        stacking_classifier = joblib.load(classifier_path)
    else:
        # Train the stacking classifier if it doesn't exist
        stacking_classifier.fit(X_train_tfidf, y_train)
        # Save the trained classifier
        joblib.dump(stacking_classifier, classifier_path)

    # Train individual classifiers for accuracy calculation
    nb_classifier.fit(X_train_tfidf, y_train)
    rf_classifier.fit(X_train_tfidf, y_train)
    svm_classifier.fit(X_train_tfidf, y_train)
    dt_classifier.fit(X_train_tfidf, y_train)
    knn_classifier.fit(X_train_tfidf, y_train)
    lr_classifier.fit(X_train_tfidf, y_train)

    # Calculate accuracy for individual models
    individual_accuracies = {
        'nb': accuracy_score(y_test, nb_classifier.predict(X_test_tfidf)),
        'rf': accuracy_score(y_test, rf_classifier.predict(X_test_tfidf)),
        'svm': accuracy_score(y_test, svm_classifier.predict(X_test_tfidf)),
        'dt': accuracy_score(y_test, dt_classifier.predict(X_test_tfidf)),
        'knn': accuracy_score(y_test, knn_classifier.predict(X_test_tfidf)),
        'lr': accuracy_score(y_test, lr_classifier.predict(X_test_tfidf))
    }

    # Calculate accuracy for the hybrid model
    hybrid_accuracy = accuracy_score(y_test, stacking_classifier.predict(X_test_tfidf))

    return individual_accuracies, hybrid_accuracy

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    # Check if the file has a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file to a temporary directory
    file_path = os.path.join('uploads', secure_filename(file.filename))
    file.save(file_path)

    # Load and preprocess the data
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_data(file_path)

    # Train the classifiers and get accuracies
    individual_accuracies, hybrid_accuracy = train_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, file_path)

    # Clean up: remove the uploaded file
    os.remove(file_path)

    # Redirect to a different URL after processing the request
    return redirect(url_for('show_result', individual_accuracies=individual_accuracies, hybrid_accuracy=hybrid_accuracy))

@app.route('/show_result')
def show_result():
    individual_accuracies = request.args.get('individual_accuracies', None)
    hybrid_accuracy = request.args.get('hybrid_accuracy', None)

    # Convert individual_accuracies string to dictionary
    individual_accuracies = eval(individual_accuracies)

    # Add hybrid accuracy to individual accuracies dictionary
    individual_accuracies['stacking'] = float(hybrid_accuracy)

    # Mapping of model names to display names
    model_names_mapping = {
        'nb': 'Naive Bayes',
        'svm': 'SVM',
        'dt': 'Decision Tree',
        'rf': 'Random Forest',
        'knn': 'KNN',
        'lr': 'Logistic Regression',
        'stacking': 'Hybrid Model'
    }

    # Find the model with the highest accuracy
    best_model = max(individual_accuracies, key=individual_accuracies.get)
    best_model_display_name = model_names_mapping.get(best_model, best_model)

    return render_template('show_result.html', individual_accuracies=individual_accuracies, best_model=best_model_display_name, model_names_mapping=model_names_mapping)



if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

