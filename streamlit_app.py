import streamlit as st
import pandas as pd
from AAM_long_toolbox import plot_network_with_lf_res, generate_curves, \
    create_population_estimations,train_xgboost_model,make_predictions, model_comparison
import os


def check_csv_structure(file_path, required_columns):
    df = pd.read_csv(file_path)
    actual_columns = df.columns.tolist()

    missing_required = [col for col in required_columns if col not in actual_columns]

    if missing_required:
        st.error(f"❌ Missing required columns: {missing_required}")
        st.stop()

    if not missing_required:
        st.success(f"✅ Input file has the correct format")
        return df

st.set_page_config(layout='wide')

if 'cdfs' not in st.session_state:
    st.session_state.cdfs = None


required_csv_columns = ["smart_meter_id", "delivery_point_id", "brand", "model", "installation_date"]

# Set the title of the Streamlit app
st.title("Long Term Asset Management")
if st.session_state.cdfs is not None:
    tab = st.sidebar.radio("Tabs", ["Import Smart Meter Data & Failure Logs", "End of Life Curves Dashboard", "Critical Smart Meters Dashboard"])
else:
    tab = st.sidebar.radio("Tabs", ["Import Smart Meter Data & Failure Logs"])


## Define the tabs
#tabs = st.tabs(["Historical Data Input","End of Life Curves","Critical Meters Calculation","Map"])

# Content for the 'Home' tab
if tab == "Import Smart Meter Data & Failure Logs":
    st.header("Historical Data Input")
    st.write("Provide Meter Installation & Failure Dates")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"],key=1)

    # Process the file after it's uploaded
    if uploaded_file is not None:
        # Read the CSV file and generate curves
        if uploaded_file.name.endswith("csv"):
            if st.session_state.cdfs is None:
                df = check_csv_structure(uploaded_file, required_csv_columns)
                st.write("File content as DataFrame:")
                st.write(df)
                # Run generate_curves only once per upload
                status_placeholder = st.empty()
                status_placeholder.write("🔄 Generating End of Life curves...")
                Meter_data, cdfs = generate_curves(df)
                st.session_state.Meter_data = Meter_data
                st.session_state.cdfs = cdfs
                status_placeholder.success("✅ End of Life curves generated.")
        else:
            st.write("File is not csv")
    else:
        st.session_state.cdfs = None
        st.session_state.Meter_data = None



if tab == "End of Life Curves Dashboard":
    cols = st.columns([50, 50])
    with cols[0]:
        st.subheader("End of life curves per model")
        if st.session_state.cdfs is not None:
            brands = st.session_state.Meter_data.brand.unique()
            selected_brand =st.selectbox("Select a Smart Meter brand:", brands)
            models = st.session_state.Meter_data[st.session_state.Meter_data.brand==selected_brand].model.unique()
            selected_model = st.selectbox("Select a Smart Meter brand:", models)
            if os.path.exists('Plots/EOL/'+str(selected_model)+'_EOL_curve.html'):
                with open('Plots/EOL/'+str(selected_model)+'_EOL_curve.html', 'r', encoding='utf-8') as html_file:
                    plot_html = html_file.read()
                st.components.v1.html(plot_html, height=600)
            else:
                st.write("Limited failure numbers recorded")
        else:
            st.write("Update Historical Data File")
    with cols[1]:
        st.subheader("Model Comparison")
        st.write("Smart Meters models that have reached at least 5% failure rate are compared to the time required to reach 20% failure rate")
        st.dataframe(model_comparison(st.session_state.cdfs,st.session_state.Meter_data))


    st.subheader("Future Predicted Failures")
    # Request an integer value from the month ahead
    month_value = st.number_input("Months ahead:",
                                    min_value=0,  # Set the minimum value
                                    max_value=24,  # Set the maximum value
                                    value=1,  # Default starting value
                                    step=1)  # Increment step
    if st.session_state.cdfs is not None:
        Expected_Failures = create_population_estimations(st.session_state.Meter_data, st.session_state.cdfs,month_value)
        st.write(Expected_Failures)
#
# with tabs[2]:
#     st.header("Critical Meter Calculation")
#     st.write("Provide Meter Data")
#
#     # File uploader
#     uploaded_file2 = st.file_uploader("Choose a file", type=["csv"],key=2)
#
#     # Process the file after it's uploaded
#     if uploaded_file2 is not None:
#         if 'uploaded_file2' not in st.session_state or st.session_state['uploaded_file2'] != uploaded_file2.name:
#             # Store the uploaded file name in session state
#
#             # Read the CSV file and generate curves
#             if uploaded_file2.name.endswith("csv"):
#                 DATA = pd.read_csv(uploaded_file2)
#                 st.session_state['uploaded_file2'] = uploaded_file2.name
#                 st.session_state['DATA'] = DATA
#
#                 # Train the model
#                 train_xgboost_model(DATA)
#             else:
#                 st.write("File is not csv")
#         else:
#             # If file is already uploaded, display the previous result from session state
#             DATA = st.session_state.get('DATA', None)
#             #st.write("File content as DataFrame:")
#             #st.write(DATA)
#     else:
#         st.write("Please upload a file to see the content.")
#
#     if 'uploaded_file2'  in st.session_state:
#         if 'probability_SM' not in st.session_state:
#             probability_SM = make_predictions(DATA, cdfs)
#             st.session_state['probability_SM'] = probability_SM
#         else:
#             probability_SM =  st.session_state.get('probability_SM', None)
#         st.write('Critical Smart Meters')
#         st.write(probability_SM[probability_SM['Failure Probability'] >= 70])
#         with open("feature_importance_plot.html", 'r', encoding='utf-8') as file:
#             html_content = file.read()
#         st.components.v1.html(html_content, width=1000, height=400, scrolling=True)
#
# with tabs[3]:
#     st.header("Map")
#     st.write("Upload Substation Topology File")
#
#     # File uploader
#     topology_file = st.file_uploader("Choose a file", type=["csv"],key=3)
#     if topology_file is not None:
#         if 'topology_file' not in st.session_state:
#             if topology_file.name.endswith("csv"):
#                 st.session_state['topology_file'] = topology_file.name
#             else:
#                 st.write("File is not csv")
#         else:
#             topology_file = st.session_state.get('topology_file', None)
#
#         if ('topology' not in st.session_state):
#             topology = pd.read_csv(topology_file,delimiter=';')
#             st.session_state['topology'] = topology
#         else:
#             topology = st.session_state.get('topology', None)
#     else:
#         st.write("Upload Topology File in csv.")
#     if ('probability_SM' in st.session_state)&('topology' in st.session_state):
#         plot_network_with_lf_res(topology, probability_SM[probability_SM['Failure Probability'] >= 70])
#         with open("network_map.html", 'r', encoding='utf-8') as file:
#             html_content = file.read()
#         st.components.v1.html(html_content, width=1000, height=400, scrolling=True)
