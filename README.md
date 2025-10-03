# Sentiment Analysis on Instrument Reviews

This project performs **sentiment analysis** on customer reviews of musical instruments using Python and machine learning techniques.

## Project Overview
- Data: `Instruments_Reviews.csv` containing review text, summary, ratings, and other metadata.
- Goal: Classify reviews as **Positive**, **Negative**, or **Neutral** based on customer ratings and review text.

## Key Steps
1. **Data Preprocessing**
   - Combine review text and summary into a single column.
   - Handle missing values and drop irrelevant columns.
   - Clean text: lowercase, remove punctuation, links, repeated characters, numbers.
   - Tokenize, remove stopwords, and lemmatize text using NLTK.

2. **Feature Engineering**
   - Transform reviews into **TF-IDF vectors** (bigrams, max 5000 features).
   - Encode sentiment labels.
   - Balance classes using **SMOTE**.

3. **Modeling**
   - Split data into training and test sets.
   - Train an **SVM classifier** with hyperparameter tuning using **GridSearchCV**.
   - Evaluate model performance with **cross-validation** and classification metrics.

4. **Results**
   - Best SVM parameters: `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}`
   - Cross-validation F1 macro score: ~0.973
   - Test set accuracy: ~97%
   - Model shows strong precision and recall for all sentiment classes.

## Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk` for text processing
- `scikit-learn` for machine learning
- `imblearn` for SMOTE oversampling
