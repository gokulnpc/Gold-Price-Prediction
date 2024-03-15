import streamlit as st
import pandas as pd
import pickle
import joblib

# Function to load the model
@st.cache_data
def load_model():
    with open('gold_price_prediction_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# with open('saved _model/rf_model', 'rb') as file:
#     loaded_model = pickle.load(file)

# Load your model
loaded_model = load_model()

# Function to create the input datafram
def create_input_df(user_inputs):
    input_df = pd.DataFrame([user_inputs])
    return input_df

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Gold Price Prediction Web App')

    
    # User inputs: ['SPX', 'USO', 'SLV', 'EUR/USD']
    spx = st.number_input('SPX', value=0.0)
    uso = st.number_input('USO', value=0.0)
    slv = st.number_input('SLV', value=0.0) 
    eur_usd = st.number_input('EUR/USD', value=0.0)
    

    user_inputs = { 
        'SPX': spx,
        'USO': uso,
        'SLV': slv,
        'EUR/USD': eur_usd
    }

    if st.button('Predict'):
        input_df = create_input_df(user_inputs)
        prediction = loaded_model.predict(input_df)
        st.markdown(f'**The predicted Gold price is: {prediction[0]:,.2f}**')  # Display prediction with bold
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Random Forest Regressor')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'gold_price_prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="gold_price_prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'gld_price_data.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="gld_price_data.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Gold-Price-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is a simple Gold price prediction web app. The web app uses a Random Forest Regressor model to predict the Gold price based on the user inputs.')
    st.write('--'*50)

    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Gold-Price-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
