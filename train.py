import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Загрузка данных с Kaggle
df = pd.read_csv('train.csv')

# 2. Выбор полезных признаков и целевой переменной
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].copy()
y = df['Survived']

# 3. Обработка пропущенных значений и кодирование категорий
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 4. Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Сохранение модели
joblib.dump(model, 'model.pkl')
print("Модель успешно обучена и сохранена как model.pkl!")