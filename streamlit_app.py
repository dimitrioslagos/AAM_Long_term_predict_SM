import streamlit as st
import pandas as pd
from AAM_long_toolbox import plot_network_with_lf_res, generate_curves, create_population_estimations,train_xgboost_model,make_predictions
import os

# Set the title of the Streamlit app
st.title("Long Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input","End of Life Curves","Critical Meters Calculation","Map"])

# Content for the 'Home' tab
with tabs[0]:
    st.header("Historical Data Input")
    st.write("Provide Meter Installation & Failure Dates")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"],key=1)

    # Process the file after it's uploaded
    if uploaded_file is not None:
        if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != uploaded_file.name:
            # Store the uploaded file name in session state

            # Read the CSV file and generate curves
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
                st.session_state['uploaded_file'] = uploaded_file.name
                st.write("File content as DataFrame:")
                st.write(df)

                # Run generate_curves only once per upload
                Meter_data, cdfs = generate_curves(df)
                st.session_state['Meter_data'] = Meter_data
                st.session_state['cdfs'] = cdfs # Store the result in session state
            else:
                st.write("File is not csv")
        else:
            # If file is already uploaded, display the previous result from session state
            Meter_data = st.session_state.get('Meter_data', None)
            cdfs = st.session_state.get('cdfs', None)


    else:
        st.write("Please upload a file to see the content.")

with tabs[1]:
    st.subheader("End of life curves per model")
    if uploaded_file is not None:
        brands = Meter_data.brand.unique()
        selected_brand =st.selectbox("Select a Smart Meter brand:", brands)
        models = Meter_data[Meter_data.brand==selected_brand].model.unique()
        selected_model = st.selectbox("Select a Smart Meter brand:", models)
        if os.path.exists('Plots/EOL/'+str(selected_model)+'_EOL_curve.html'):
            with open('Plots/EOL/'+str(selected_model)+'_EOL_curve.html', 'r', encoding='utf-8') as html_file:
                plot_html = html_file.read()
            st.components.v1.html(plot_html, height=600)
        else:
            st.write("Limited failure numbers recorded")
    else:
        st.write("Update Historical Data File")
    ####
    st.subheader("Future Predicted Failures")
    # Request an integer value from the month ahead
    month_value = st.number_input("Months ahead:",
                                    min_value=0,  # Set the minimum value
                                    max_value=24,  # Set the maximum value
                                    value=1,  # Default starting value
                                    step=1)  # Increment step
    if 'uploaded_file'  in st.session_state:
        Expected_Failures = create_population_estimations(Meter_data, cdfs,month_value)
        st.write(Expected_Failures)

with tabs[2]:
    st.header("Critical Meter Calculation")
    st.write("Provide Meter Data")

    # File uploader
    uploaded_file2 = st.file_uploader("Choose a file", type=["csv"],key=2)

    # Process the file after it's uploaded
    if uploaded_file2 is not None:
        if 'uploaded_file2' not in st.session_state or st.session_state['uploaded_file2'] != uploaded_file2.name:
            # Store the uploaded file name in session state

            # Read the CSV file and generate curves
            if uploaded_file2.name.endswith("csv"):
                DATA = pd.read_csv(uploaded_file2)
                st.session_state['uploaded_file2'] = uploaded_file2.name
                st.session_state['DATA'] = DATA

                # Train the model
                train_xgboost_model(DATA)
            else:
                st.write("File is not csv")
        else:
            # If file is already uploaded, display the previous result from session state
            DATA = st.session_state.get('DATA', None)
            st.write("File content as DataFrame:")
            st.write(DATA)
    else:
        st.write("Please upload a file to see the content.")

    if 'uploaded_file2'  in st.session_state:
        if 'probability_SM' not in st.session_state:
            probability_SM = make_predictions(DATA, cdfs)
            st.session_state['probability_SM'] = probability_SM
        else:
            probability_SM =  st.session_state.get('probability_SM', None)
        st.write('Critical Smart Meters')
        st.write(probability_SM[probability_SM['Failure Probability'] >= 70])

with tabs[3]:
    st.header("Map")
    st.write("Upload Substation Topology File")

    # File uploader
    topology_file = st.file_uploader("Choose a file", type=["csv"],key=3)
    if topology_file is not None:
        if 'topology_file' not in st.session_state:
            if topology_file.name.endswith("csv"):
                st.session_state['topology_file'] = topology_file.name
            else:
                st.write("File is not csv")
        else:
            topology_file = st.session_state.get('topology_file', None)

        if ('topology' not in st.session_state)
            topology = pd.read_csv('topology_substation.csv',delimiter=';')
            st.session_state['topology'] = topology
        else:
            topology = st.session_state.get('topology', None)
    else:
        st.write("Upload Topology File in csv.")
    if ('probability_SM' in st.session_state)&('topology' in st.session_state):
        plot_network_with_lf_res(topology, probability_SM[probability_SM['Failure Probability'] >= 70])
