# Task 5: Auto Tagging Support Tickets Using LLM
## Objective
Automatically classify customer support tickets into categories using a Large Language Model (LLM).
## Dataset
support_tickets.csv
## Features
- Zero-shot classification using Facebook BART Large MNLI
- Few-shot learning
- Top 3 predicted tags
- Accuracy comparison
- F1-score evaluation
- Streamlit web application
## Technologies Used
- Python
- Transformers
- Hugging Face
- PyTorch
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
## Project Structure
```
Task5_Auto_Tagging_Support_Tickets_LLM
│
├── data
│   └── support_tickets.csv
│
├── results
│   ├── zero_shot_predictions.csv
│   ├── few_shot_predictions.csv
│   ├── performance_comparison.csv
│   ├── performance_comparison.png
│   └── streamlit_output.png
│
├── zero_shot.py
├── few_shot.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```
## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the Streamlit application:
```bash
streamlit run app.py
```
## Results
- Zero-shot Accuracy: **0.40**
- Few-shot Accuracy: **0.40**
- Zero-shot F1 Score: **0.3167**
- Few-shot F1 Score: **0.40**
## Author
Sharfa Nawab