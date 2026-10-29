Amazon Product Reviews Analytics :

Project Overview :

This project analyzes Amazon customer reviews using Python, NLP, Machine Learning, and Streamlit.
The notebook provided for the project performs data cleaning, sentiment creation from ratings, NLP text preprocessing, EDA, TF-IDF vectorization, Logistic Regression classification, model evaluation, and Streamlit deployment.

Dataset :

The supplied dataset contains 21,214 rows and 9 original columns:

Reviewer Name

Profile Link

Country

Review Count

Review Date

Rating

Review Title

Review Text

Date of Experience

After duplicate removal and removal of rows without review text, the working dataset used for the analysis contains 21,055 rows.

Sentiment Logic :

Rating 4 or 5 -> Positive

Rating 3 -> Neutral

Rating 1 or 2 -> Negative

Technologies :

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, Joblib, Streamlit.

Machine Learning Workflow :

Load the CSV dataset.

Clean column names and remove duplicate/missing review records.

Extract the numeric rating.

Create the sentiment target.

Clean review text.

Split data into training and testing sets (80/20).

Convert text to TF-IDF features with up to 5,000 features.

Train Logistic Regression.

Evaluate using accuracy, classification report, and confusion matrix.

Save the trained model and vectorizer.

Use Streamlit for interactive sentiment prediction.

Observed Model Result:

Using the supplied dataset and the notebook's Logistic Regression + TF-IDF approach, the reproduced 80/20 test split achieved approximately 89.48% accuracy.
The Neutral class has much lower recall than the Positive and Negative classes, so the model should not be interpreted as equally strong for all three classes.

Project Files:

Amazon.capstone/
├── app.py
├── Amazon_Reviews.csv
├── Amazon_Reviews_Analytics.ipynb
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── Project_Report.docx
└── Amazon_Reviews_Analytics.pptx

Installation :

Open the project folder in VS Code Terminal:

python -m pip install -r requirements.txt

NLTK Downloads:

Run once in Python/Jupyter if your preprocessing uses NLTK stopwords and WordNet:

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

Run the Notebook:

Open Amazon_Reviews_Analytics.ipynb in VS Code or Jupyter and run the cells in order.

Save the Model :

The notebook uses Joblib:

import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

Run Streamlit :

Make sure app.py, model.pkl, and vectorizer.pkl are in the same project folder.

streamlit run app.py

Then open the local Streamlit URL shown in the terminal, normally:
http://localhost:8501/

Important File-Name Note :

Your uploaded dataset is named Amazon(2).csv. If your code expects Amazon_Reviews.csv, either rename the CSV to Amazon_Reviews.csv or change the filename inside the Python code.

Limitations :

The Neutral class is much smaller than the Negative class. Future work can address class imbalance, compare additional models, tune hyperparameters, and improve the Streamlit dashboard with more analytics.