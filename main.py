import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #D3C2CF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache(allow_output_mutation=True) #load the data just once
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data

with header:
    st.title('Welcome to my project!')
    st.text('In this project I look into the transactions of taxis in NYC')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on blablabla.com')

    taxi_data = get_data('taxi_data.csv')
    #st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pulocation_disp = pd.DataFrame(taxi_data['puLocationId'].value_counts()).head(50)
    st.bar_chart(pulocation_disp)

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic...')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the parameters of the model and see how performance change')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value = 10, max_value = 100, value = 20, step = 10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options = [100, 200, 300, 'No limit'], index = 0)

    sel_col.text('Here is the list of features that you can choose: ')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'puLocationId')

    # Handling missing values with mean imputation
    taxi_data.fillna(taxi_data.mean(), inplace=True) 

    #regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth = max_depth)
    else:
        regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['tripDistance']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))
