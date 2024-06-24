import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the trained models
model_lr = pickle.load(open('model_lr.sav', 'rb'))
model_nb = pickle.load(open('model_nb.sav', 'rb'))
model_nn = pickle.load(open('model_nn.sav', 'rb'))

# Function to normalize new data
def normalize_data(new_data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(new_data)

# Streamlit app
st.title("Aplikasi Prediksi Stunting")

menu = ["Dashboard", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Dashboard":
    st.subheader("Dashboard")
    st.write("Selamat datang di aplikasi prediksi stunting.")

elif choice == "Predict":
    st.subheader("Predict")
    st.write("Masukkan data untuk melakukan prediksi:")

    Berat = st.text_input('Input nilai Berat', key='berat_input')
    Tinggi = st.text_input('Input nilai Tinggi', key='tinggi_input')
    BB_U = st.text_input('Input nilai BB/U', key='bb_u_input')
    TB_U = st.text_input('Input nilai TB/U', key='tb_u_input')
    JK = st.text_input('Input Jenis kelamin (0 = Laki-laki, 1 = Perempuan)', key='jk_input')

    predict_pred_lr = ''
    predict_pred_nb = ''
    predict_pred_nn = ''

    if st.button('Test prediksi stunting'):
        if Berat and Tinggi and BB_U and TB_U and JK:
            # Create new data as a list of lists (excluding BB/TB)
            new_data = [[float(Berat), float(Tinggi), float(BB_U), float(TB_U), float(JK)]]
            new_data_scaled = normalize_data(new_data)

            # Predict with each model
            if model_lr.coef_.shape[0] == 5:  # Check if the number of features matches
                predict_pred_lr = model_lr.predict(new_data_scaled)

            if model_nb.__class__.__name__ == 'GaussianNB':  # Example check for Naive Bayes (if applicable)
                predict_pred_nb = model_nb.predict(new_data_scaled)

            if model_nn.input_shape[1] == 5:  # Example check for Neural Network (if applicable)
                predict_pred_nn = model_nn.predict(new_data_scaled)

    st.write("Hasil Prediksi:")
    st.write(f"Prediksi dengan Linear Regression: {predict_pred_lr}")
    st.write(f"Prediksi dengan Naive Bayes: {predict_pred_nb}")
    st.write(f"Prediksi dengan Neural Network: {predict_pred_nn}")





# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import pickle

# # Load the trained models
# model_lr = pickle.load(open('model_lr.sav', 'rb'))
# model_nb = pickle.load(open('model_nb.sav', 'rb'))
# model_nn = pickle.load(open('model_nn.sav', 'rb'))

# # Function to normalize new data
# def normalize_data(new_data):
#     scaler = MinMaxScaler()
#     return scaler.fit_transform(new_data)

# # Streamlit app
# st.title("Aplikasi Prediksi Stunting")

# menu = ["Dashboard", "Predict"]
# choice = st.sidebar.selectbox("Menu", menu)

# if choice == "Dashboard":
#     st.subheader("Dashboard")
#     st.write("Selamat datang di aplikasi prediksi stunting.")

# elif choice == "Predict":
#     st.subheader("Predict")
#     st.write("Masukkan data untuk melakukan prediksi:")

#     Berat = st.text_input('Input nilai Berat', key='berat_input')
#     Tinggi = st.text_input('Input nilai Tinggi', key='tinggi_input')
#     BB_U = st.text_input('Input nilai BB/U', key='bb_u_input')
#     TB_U = st.text_input('Input nilai TB/U', key='tb_u_input')
#     JK = st.text_input('Input Jenis kelamin (0 = Laki-laki, 1 = Perempuan)', key='JK_input')

#     predict_pred_lr = ''
#     predict_pred_nb = ''
#     predict_pred_nn = ''

#     if st.button('Test prediksi stunting'):
#         if Berat and Tinggi and BB_U and TB_U and JK:
#             # Create new data as a list of lists
#             new_data = [[float(Berat), float(Tinggi), float(BB_U), float(TB_U), float(JK)]]
#             new_data_scaled = normalize_data(new_data)

#             # Predict with each model
#             if model_lr.coef_.shape[0] == 6:  # Check if the number of features matches
#                 predict_pred_lr = model_lr.predict(new_data_scaled)

#             if model_nb.n_features_ == 6:  # Example check for Naive Bayes (if applicable)
#                 predict_pred_nb = model_nb.predict(new_data_scaled)

#             if model_nn.input_shape[1] == 6:  # Example check for Neural Network (if applicable)
#                 predict_pred_nn = model_nn.predict(new_data_scaled)

#     st.write("Hasil Prediksi:")
#     st.write(f"Prediksi dengan Linear Regression: {predict_pred_lr}")
#     st.write(f"Prediksi dengan Naive Bayes: {predict_pred_nb}")
#     st.write(f"Prediksi dengan Neural Network: {predict_pred_nn}")
