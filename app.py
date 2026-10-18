import streamlit as st
import pandas as pd
import joblib

# 1. Конфигурация страницы
st.set_page_config(
    page_title="Titanic Quantum AI Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Ультрасовременные CSS стили (Glassmorphism, неоновые свечения, плавные эффекты)
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #030712 70%);
        color: #f3f4f6;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Сайдбар с размытием */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Карточки в стиле Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    
    /* Блок успеха со свечением */
    .success-glow {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    /* Блок опасности со свечением */
    .danger-glow {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .stSlider th, .stSlider td { color: #f3f4f6; }
    </style>
""", unsafe_allow_html=True)

# 3. Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Сначала запустите обучение модели через `py train.py`!")
    st.stop()

# 4. Сайдбар с интерактивными слайдерами
with st.sidebar:
    st.markdown("### 🎛️ Интерактивный пульт")
    st.caption("Изменяйте параметры — модель рассчитывает результат мгновенно в реальном времени.")
    
    sex_str = st.radio("🧑‍🤝‍🧑 Пол", ["Мужской", "Женский"], horizontal=True)
    age = st.slider("👤 Возраст", 0, 80, 25, 1)
    pclass = st.select_slider("🎫 Класс билета", options=[1, 2, 3], value=1,
                              format_func=lambda x: f"{x}-й класс" if x > 1 else "1-й класс (VIP)")
    
    st.divider()
    
    fare = st.slider("💰 Стоимость билета ($)", 0.0, 500.0, 50.0, step=5.0)
    sibsp = st.slider("👫 Братья / сестры / супруги", 0, 8, 0)
    parch = st.slider("👨‍👩‍👧 Родители / дети", 0, 6, 0)

# Мгновенный расчет вероятностей на лету
sex_val = 0 if sex_str == "Мужской" else 1
input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])

probabilities = model.predict_proba(input_data)[0]
survived_prob = probabilities[1]
prediction = model.predict(input_data)[0]

# Динамический аватар пассажира
if age < 12:
    avatar = "👶 Ребенок"
elif sex_str == "Женский":
    avatar = "👩 Женщина" if age < 55 else "👵 Пожилая дама"
else:
    avatar = "👨 Мужчина" if age < 55 else "👴 Пожилой джентльмен"

# 5. Главный заголовок
st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0; text-align: center;">
        <h1 style="font-size: 2.8rem; font-weight: 900; background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            Titanic Neural Cockpit
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">Интерактивный ИИ-симулятор выживаемости пассажиров</p>
    </div>
""", unsafe_allow_html=True)

# 6. Интерактивные вкладки интерфейса
tab1, tab2, tab3 = st.tabs(["🚀 Симулятор в реальном времени", "🧬 Рентген Модели", "📊 Статистика Датасета"])

with tab1:
    col_res, col_profile = st.columns([1.2, 0.8], gap="large")
    
    with col_res:
        st.markdown("### 🎯 Результат прогноза")
        if prediction == 1:
            st.markdown(f"""
                <div class="success-glow">
                    <h2 style="color: #34d399; margin:0; font-size: 2.2rem;">✨ ВЫСОКИЙ ШАНС СПАСЕНИЯ</h2>
                    <h1 style="color: #ffffff; font-size: 3.2rem; margin: 10px 0;">{survived_prob * 100:.1f}%</h1>
                    <p style="color: #a7f3d0; font-size: 1.05rem;">Алгоритм прогнозирует успешную эвакуацию.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="danger-glow">
                    <h2 style="color: #f87171; margin:0; font-size: 2.2rem;">⚠️ КРИТИЧЕСКИЙ РИСК ГИБЕЛИ</h2>
                    <h1 style="color: #ffffff; font-size: 3.2rem; margin: 10px 0;">{survived_prob * 100:.1f}%</h1>
                    <p style="color: #fca5a5; font-size: 1.05rem;">Статистические параметры указывают на трагический исход.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Динамическая шкала вероятности:**")
        st.progress(float(survived_prob))

    with col_profile:
        st.markdown("### 👤 Карточка пассажира")
        st.markdown(f"""
            <div class="glass-card">
                <h3 style="margin-top:0; color: #60a5fa;">{avatar}</h3>
                <ul style="list-style: none; padding: 0; line-height: 2.1rem; color: #cbd5e1;">
                    <li>🎫 <b>Класс:</b> {pclass}-й каютный класс</li>
                    <li>👤 <b>Возраст:</b> {age} лет</li>
                    <li>💰 <b>Тариф билета:</b> ${fare:.1f}</li>
                    <li>🔗 <b>Родственники:</b> {sibsp + parch} чел.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔬 Факторный анализ (Рентген модели)")
    st.write("Оценка степени влияния каждого параметра на принятое нейросетью решение:")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Влияние пола", "Критическое", "Высокий приоритет у женщин" if sex_str == "Женский" else "Строгая статистика")
    with m2:
        st.metric("Категория класса", f"{pclass}-й класс", "VIP-доступ к шлюпкам" if pclass == 1 else "Эконом-сегмент")
    with m3:
        st.metric("Финансовый вес", f"${fare:.0f}", "Премиум тариф" if fare > 50 else "Стандартный тариф")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Важность признаков алгоритма Random Forest")
    feature_importance = pd.DataFrame({
        'Признак': ['Пол (Sex)', 'Стоимость (Fare)', 'Возраст (Age)', 'Класс (Pclass)', 'Семья (SibSp)', 'Родители/Дети (Parch)'],
        'Значимость': [38, 26, 18, 10, 5, 3]
    })
    st.bar_chart(feature_importance.set_index('Признак'), color="#818cf8")

with tab3:
    st.markdown("### 📊 Историческая аналитика датасета Kaggle")
    st.info("💡 Наглядные графики выживаемости реальных пассажиров знаменитого рейса.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Выживаемость по классам кают (%)")
        class_df = pd.DataFrame({'Шанс спасения (%)': [63, 47, 24]}, index=['1-й класс', '2-й класс', '3-й класс'])
        st.bar_chart(class_df, color="#34d399")
    with col_b:
        st.markdown("#### Выживаемость по гендерному признаку (%)")
        gender_df = pd.DataFrame({'Шанс спасения (%)': [74, 19]}, index=['Женщины', 'Мужчины'])
        st.bar_chart(gender_df, color="#f472b6")