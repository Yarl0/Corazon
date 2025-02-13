import streamlit as st
import joblib
import pandas as pd

# Título de la aplicación
st.title("Jojos ataque al corazón")

# Introducción
st.write("""
Esta aplicación predice si una persona sufre o no de problemas cardíacos utilizando un modelo de KNN entrenado con datos de edad y colesterol.
""")

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Crear pestañas
tab1, tab2 = st.tabs(["Instrucciones y Entrada de Datos", "Predicción"])

with tab1:
    st.header("Instrucciones y Entrada de Datos")
    st.write("""
    Por favor, ingresa los siguientes datos para realizar la predicción:
    - **Edad**: Entre 18 y 80 años.
    - **Colesterol**: Entre 50 y 600.
    """)
    
    # Entrada de datos
    edad = st.slider("Edad", min_value=18, max_value=80, value=50)
    colesterol = st.slider("Colesterol", min_value=50, max_value=600, value=200)

    # Crear un DataFrame con los datos de entrada
    datos_usuario = pd.DataFrame({
        'edad': [edad],
        'colesterol': [colesterol]
    })

    # Normalizar los datos de entrada
    datos_usuario_escalados = escalador.transform(datos_usuario)

with tab2:
    st.header("Predicción")
    
    # Realizar la predicción
    prediccion = modelo_knn.predict(datos_usuario_escalados)
    
    # Mostrar el resultado de la predicción
    if prediccion[0] == 0:
        st.success("No presenta problemas cardíacos.")
    else:
        st.error("Presenta problemas cardíacos.")
        st.image("https://i.pinimg.com/564x/a5/36/a2/a536a2f63ba3ccc7ed0cfe081335a9ef.jpg", caption="Problema cardíaco detectado.")

# Para ejecutar la aplicación, usa el siguiente comando en la terminal:
# streamlit run app.py
