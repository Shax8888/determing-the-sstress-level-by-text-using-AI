from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys
import traceback

# Расширенный английский датасет (70 примеров: 35 HIGH, 35 LOW)
texts = [
    # HIGH stress
    "I’m so anxious, I can’t cope anymore.",
    "Everything is falling apart, I’m overwhelmed.",
    "I’m stressed out and can’t sleep at night.",
    "Work is killing me, I’m so tired.",
    "I feel like I’m drowning in problems.",
    "I can’t handle this pressure anymore.",
    "My heart is racing, I’m so nervous.",
    "I’m constantly worried about everything.",
    "I’m under so much pressure, I can’t think straight.",
    "I feel trapped and don’t know what to do.",
    "I’m exhausted from all this stress.",
    "I can’t stop thinking about my problems.",
    "I feel like I’m on the edge of a breakdown.",
    "This workload is crushing me.",
    "I’m so tense, I can’t relax at all.",
    "I’m freaking out about my deadlines.",
    "I feel hopeless and overwhelmed.",
    "I’m so nervous, my hands are shaking.",
    "I can’t deal with all these responsibilities.",
    "I’m stressed to the point of tears.",
    "I feel like everything is going wrong.",
    "I’m so worried, I can’t focus on anything.",
    "This stress is making me sick.",
    "I’m panicking about what’s next.",
    "I feel like I’m losing control.",
    "I’m terrified about my upcoming exams.",
    "I’m so stressed, I can’t concentrate.",
    "I feel like I’m failing at everything.",
    "I’m overwhelmed by all these tasks.",
    "I’m anxious about what the future holds.",
    "I’m scared I won’t meet my deadlines.",
    "I feel suffocated by all this pressure.",
    "I can’t stop feeling nervous all the time.",
    "I’m so tired of being stressed out.",
    "I feel like I’m about to break down.",

    # LOW stress
    "Today is a great day, I’m full of energy!",
    "I’m calm and happy with how things are going.",
    "Life is good, I’m enjoying every moment.",
    "I feel relaxed and at peace today.",
    "Everything is under control, I’m content.",
    "I had a wonderful day with friends.",
    "I’m excited about my new project!",
    "Feeling refreshed after a good night’s sleep.",
    "I’m looking forward to a relaxing weekend.",
    "I’m in a great mood today!",
    "I feel optimistic and ready for anything.",
    "Today was productive and stress-free.",
    "I’m so happy with how things turned out.",
    "I feel light and carefree today.",
    "I enjoyed a relaxing evening at home.",
    "I’m grateful for this peaceful moment.",
    "I’m energized and motivated today.",
    "I had a fantastic time with my family.",
    "I feel confident and at ease.",
    "Life feels balanced and calm right now.",
    "I’m thrilled about my recent success.",
    "I’m enjoying this sunny, cheerful day.",
    "I feel rested and full of positivity.",
    "I’m content with where I am in life.",
    "I’m smiling because today is amazing!",
    "I’m loving this beautiful weather today.",
    "I’m feeling optimistic and full of energy.",
    "I had a great workout and feel amazing.",
    "I’m excited for what’s coming next.",
    "I feel so relaxed after my vacation.",
    "I’m happy to be spending time outdoors.",
    "I feel at peace with everything.",
    "I’m thrilled to start this new adventure.",
    "I had a wonderful evening with friends.",
    "I feel so refreshed and ready to go!"
]
labels = ["HIGH"] * 35 + ["LOW"] * 35

# Векторизация
try:
    print("Starting vectorization...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    print("Vectorization completed.")
except Exception as e:
    print(f"Error during vectorization: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Обучение моделей
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True, kernel='linear', C=1.0),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

for name, model in models.items():
    print(f"Training model: {name}")
    try:
        model.fit(X, labels)
        with open(f"{name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Model {name} trained and saved successfully as {name.lower().replace(' ', '_')}.pkl")
    except Exception as e:
        print(f"Error training or saving {name}: {str(e)}")
        traceback.print_exc()

try:
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Vectorizer saved successfully as vectorizer.pkl")
except Exception as e:
    print(f"Error saving vectorizer: {str(e)}")
    traceback.print_exc()

print("All models and vectorizer processing completed.")