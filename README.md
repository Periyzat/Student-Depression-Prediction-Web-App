# 🎓 Student Depression Prediction Web App 

This is a web-based application that predicts the likelihood of depression among students based on their inputs. It uses a trained Neural Network model and provides SHAP-based visual explanations for transparency.

---

## Features
- Predicts depression level using a trained ML model
- SHAP visualizations to explain prediction outcomes
- Simple and interactive web interface using Flask

---

## Built With
- Backend: Python (Flask, NumPy, Pandas, TensorFlow, SHAP)
- Frontend: HTML/CSS (Jinja templating)
- Visualization: Matplotlib, Seaborn

---

## Installation

1. Clone the Repository
```bash
git clone https://github.com/Periyzat/Student-Depression-Prediction-Web-App.git
cd Student-Depression-Prediction-Web-App
```

3. Create & Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # for Linux/macOS
venv\Scripts\activate     # for Windows
```

5. Install Dependencies
```bash
pip install -r requirements.txt
```

7. Run the Application
```bash
python app.py
```

 
## Usage

1. Open your browser and navigate to: http://127.0.0.1:5000/
2. Fill in the form with relevant personal details.
3. Submit the form to receive a prediction along with a SHAP explanation chart.

## Model Details 

- Model: Deep Neural Network (my_model.keras)
- Training Dataset: [Depression Student Dataset on Kaggle](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset)
- Explainability: SHAP values are used to interpret model decisions.

## Project Structure

```bash
Student-Depression-Prediction-Web-App/
├── app.py                   # Flask app
├── model/
│   ├── model.py             # Model loading and prediction functions
│   └── my_model.keras       # Trained deep learning model
├── dataset/
│   └── Depression Student Dataset.csv
├── templates/
│   ├── index.html           # Home page form
│   └── result.html          # Prediction + SHAP result
├── static/
│   └── depression_plot.png  # Visualization asset
├── requirements.txt         # Project dependencies
└── README.md                # Project overview
```

## Contributing
Contributions are welcome! Please open issues or submit a pull request for improvements or bug fixes.


## License
This project is licensed under the MIT License.
