# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns

# Загрузка модели (model.pkl)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']

# Загрузка данных для EDA (кэширование данных с ноутбука)
@st.cache_data
def load_eda_data():
    return pd.read_csv('df_train_for_eda.csv')
  
df_train_eda = load_eda_data() 

# Заголовок
st.title("Model for 🚗's price prediction ")

# Вкладки
tab1, tab2, tab3 = st.tabs(["📊 EDA", "⚖️Weights", "🔮 Prediction"])

 

# --- Вкладка 1: EDA ---

with tab1:
    st.header("Summarize of EDA")
    
    # Импорт изображения
    st.image("Cars.jpg") 
    
    
    # Таблица с данными
    st.subheader("Dataframe") 
    st.dataframe(df_train_eda)  


    # Построение тепловой карты на основании корреляции Пирсона
    st.subheader("Heatmap of number features")
    numeric_df = df_train_eda.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig) 
    
    # Построение гистограммы с самыми частовстречающимися марками автомобилей (топ 10)
    st.subheader("Top 10 car's brands") 
    df_train_eda['brand'] = df_train_eda['name'].str.split().str[0]
    brand_counts = df_train_eda['brand'].value_counts().head(10) 

    fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values)
    st.plotly_chart(fig) 

    # Построение распределения по топливу
    st.subheader('Type of fuel')
    fig = px.histogram(df_train_eda, x='fuel', color='fuel', title="Fuel")
    st.plotly_chart(fig)

    # Построения графика зависимости стоимости от года выпуска
    st.subheader('Selling price VS Year')
    fig = px.scatter(df_train_eda, x='year', y='selling_price',
    color = 'fuel')
    st.plotly_chart(fig)  

# --- Вкладка 2: Визуализация весов ---

with tab2:
    st.subheader("Weight's visualization")
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model.coef_
     }).sort_values('Weight', key=abs, ascending=False)
 
    
    # Или интерактивный график
    fig = px.bar(coef_df, x='Feature', y='Weight')
    st.plotly_chart(fig)



# --- Вкладка 3: Предсказание ---
with tab3:
    st.header("The price is...")
    inputs = {}
    for feat in feature_names:
        if feat == 'year':
            inputs[feat] = st.slider("Year", 1980, 2023, 2015)
        elif 'km_driven' in feat:
            inputs[feat] = st.number_input("km_driven", value=50000)
        else:
            inputs[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("Discover price"):
        X = np.array([list(inputs.values())])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Approximate price: **{pred:,.0f} $**")








 

