from flask import Flask, render_template, request
import pickle
import os
import logging
import numpy as np
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загрузка векторизатора
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    logger.info("Vectorizer loaded successfully")
except FileNotFoundError:
    logger.error("vectorizer.pkl not found. Please ensure the file exists in the project directory.")
    raise
except Exception as e:
    logger.error(f"Error loading vectorizer.pkl: {str(e)}")
    raise

# Загрузка моделей
model_files = {
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Decision Tree": "decision_tree.pkl"
}

models = {}
for name, file in model_files.items():
    if os.path.exists(file):
        try:
            with open(file, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Model {name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    else:
        logger.error(f"File {file} not found. Skipping model {name}.")

if not models:
    logger.error("No models loaded. Please ensure at least one model file exists.")
    raise FileNotFoundError("No model files found.")

@app.route("/", methods=["GET", "POST"])
def index():
    results = {}
    text = ""
    summary = ""
    if request.method == "POST":
        text = request.form["text"]
        try:
            # Предобработка текста
            text = re.sub(r'[^\w\s]', '', text.lower())
            vec = vectorizer.transform([text])
            logger.info(f"Text: {text}")
            logger.info(f"Vector shape: {vec.shape}, Non-zero elements: {vec.nnz}")
            logger.info(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")
            
            for name, model in models.items():
                prediction = model.predict(vec)[0]
                logger.info(f"{name} raw prediction: {prediction}")
                logger.info(f"{name} classes: {model.classes_}")
                
                # Translate labels (handle case sensitivity)
                prediction = str(prediction).upper()
                if prediction in ["HIGH", "1"]:
                    prediction_text = "High stress level"
                else:
                    prediction_text = "Low stress level"
                
                # Handle confidence
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec)[0]
                        # Проверим порядок классов
                        class_index = 1 if model.classes_[1] == "HIGH" else 0 if model.classes_[0] == "HIGH" else None
                        if class_index is None:
                            logger.error(f"{name} classes do not contain HIGH: {model.classes_}")
                            confidence = "Confidence calculation failed"
                        else:
                            confidence_score = proba[class_index]
                            if confidence_score >= 0.80:
                                confidence = f"High confidence ({confidence_score*100:.0f}%)"
                            elif confidence_score >= 0.60:
                                confidence = f"Medium confidence ({confidence_score*100:.0f}%)"
                            elif confidence_score >= 0.40:
                                confidence = f"Low confidence ({confidence_score*100:.0f}%)"
                            else:
                                confidence = f"Very low confidence ({confidence_score*100:.0f}%)"
                            logger.info(f"{name} probabilities: LOW={proba[0]:.2f}, HIGH={proba[1]:.2f}")
                    else:
                        confidence = "Confidence unknown"
                except Exception as e:
                    logger.error(f"Error calculating confidence for {name}: {str(e)}")
                    confidence = "Confidence calculation failed"
                
                results[name] = f"{prediction_text}. {confidence}"
            
            # Generate summary
            high_count = sum(1 for result in results.values() if "High stress level" in result)
            logger.info(f"High stress predictions: {high_count}/{len(results)}")
            if high_count > len(results) / 2:
                summary = "Most models indicate that the text expresses a high level of stress."
            elif high_count == len(results) / 2:
                summary = "Models are split on whether the text expresses stress."
            else:
                summary = "Most models indicate that the text does not express stress."
        
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            results["Error"] = f"Failed to process text: {str(e)}"
    
    return render_template("index.html", results=results, text=text, summary=summary, models=models, vectorizer=vectorizer)

if __name__ == "__main__":
    app.run(debug=True)