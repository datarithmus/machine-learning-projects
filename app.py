import streamlit as st
import joblib
import pandas as pd

st.sidebar.title('E-Bike Price Prediction')
html_temp = """
<div style="background-color:#aedcff; padding:10px;">
  <h2 style="color:#8310a9; text-align:center;">Datarithmus & Coding Book ML Solutions</h2>
</div>"""

st.markdown(html_temp, unsafe_allow_html=True)


gabel_federweg_mm=st.sidebar.selectbox("Please provide the travel length of the fork in millimeters.",( 30,
                                                                                                        35,
                                                                                                        50,
                                                                                                        60,
                                                                                                        63,
                                                                                                        65,
                                                                                                        70,
                                                                                                        75,
                                                                                                        80,
                                                                                                        100,
                                                                                                        110,
                                                                                                        120,
                                                                                                        130,
                                                                                                        140,
                                                                                                        150,
                                                                                                        160,
                                                                                                        170,
                                                                                                        180,
                                                                                                        200))
akkukapazität_wh=st.sidebar.selectbox("What is the watt-hour capacity of your electric bike's battery?", (  248,
                                                                                                            250,
                                                                                                            252,
                                                                                                            300,
                                                                                                            320,
                                                                                                            324,
                                                                                                            360,
                                                                                                            396,
                                                                                                            400,
                                                                                                            410,
                                                                                                            416,
                                                                                                            418,
                                                                                                            420,
                                                                                                            430,
                                                                                                            446,
                                                                                                            460,
                                                                                                            474,
                                                                                                            500,
                                                                                                            504,
                                                                                                            508,
                                                                                                            520,
                                                                                                            522,
                                                                                                            530,
                                                                                                            540,
                                                                                                            545,
                                                                                                            555,
                                                                                                            558,
                                                                                                            562,
                                                                                                            600,
                                                                                                            601,
                                                                                                            603,
                                                                                                            604,
                                                                                                            612,
                                                                                                            621,
                                                                                                            625,
                                                                                                            630,
                                                                                                            650,
                                                                                                            670,
                                                                                                            691,
                                                                                                            700,
                                                                                                            710,
                                                                                                            720,
                                                                                                            750,
                                                                                                            850))
rahmenmaterial=st.sidebar.radio("What material was used for the frame of your bicycle (e.g., aluminum, carbon, steel)?",('Aluminium',
                                                                                                                         'Carbon',
                                                                                                                         'Aluminium-Carbon',
                                                                                                                         'Diamant',
                                                                                                                         'Aluminium-Stahl'))
#sattel=st.sidebar.selectbox("What type of saddle do you prefer? (e.g., mountain bike saddle, road bike saddle, city bike saddle)", ('Selle Bassano Feel GT'))
gänge=st.sidebar.radio("Please specify the number of gears on your bicycle.",(0,3,5,7,8,9,10,11,12,14,20,22,24,27,30))
#bremse_vorne=st.sidebar.selectbox("Which brand of braking system located at the front of your bicycle (e.g., MAGURA HS-11 , Shimano MT-200).", ('Shimano MT-200'))
#schaltwerk=st.sidebar.selectbox("Which rear derailleur is installed on your bicycle? (e.g., Shimano Deore, SRAM GX)", ('Shimano ', 'Shimano Deore))
kategorie=st.sidebar.selectbox("To which category does your bicycle belong? (e.g., mountain bike, road bike, electric bike)", ('Trekking', 
                                                                                                                               'City',
                                                                                                                               'MTB_Hardtail',
                                                                                                                               'MTB_Fully'))
hersteller=st.sidebar.selectbox("Who is the manufacturer of your bicycle?", (   'Kalkhoff',
                                                                                'CUBE',
                                                                                'Haibike',
                                                                                'Hercules',
                                                                                'Winora',
                                                                                'SCOTT',
                                                                                'corratec',
                                                                                'Diamant',
                                                                                'GHOST',
                                                                                'Specialized',
                                                                                'Cannondale',
                                                                                'Canyon'))

rf_model=joblib.load('pipeline_model_rf')


my_dict = {
    "gabel_federweg_mm": gabel_federweg_mm,
    "akkukapazität_wh": akkukapazität_wh,
    "rahmenmaterial": rahmenmaterial,
    'gänge': gänge,
    "kategorie": kategorie,
    "hersteller": hersteller    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your e-bike is below")
st.table(df)


st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = rf_model.predict(df)
    st.success("The estimated price of your e-bike is €{}. ".format(int(prediction[0])))
