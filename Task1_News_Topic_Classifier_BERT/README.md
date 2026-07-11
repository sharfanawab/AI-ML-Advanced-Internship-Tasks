# News Topic Classifier BERT Using Machine Learning
## Project Overview
This project classifies news headlines into different topic categories using Natural Language Processing.
## Dataset
AG News Dataset
Categories:
- World
- Sports
- Business
- Technology
## Techniques Used
- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression Classification
## Model Evaluation
The model is evaluated using:
- Accuracy
- F1-score
- Confusion Matrix
## Results
Generated files:
- accuracy.png
- confusion_matrix.png
- evaluation_report.txt
## Deployment
The model is deployed using Streamlit.
Run command:
streamlit run app.py
## Project Files
- train.py : Model training
- predict.py : Single prediction
- evaluate.py : Evaluation
- app.py : Streamlit application