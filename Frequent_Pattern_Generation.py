import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from data_processing import *

st.set_page_config(layout='wide')
st.title("AdverseVis: Visual Interactive System for Adverse Behaviour Identification")

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Download CSV File</a>'
    return href

def visualize_pattern_generation(patterns_df, top20_patterns_df, algorithm_choice):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(algorithm_choice + " Algorithm")
        st.write('Data Dimension: ' + str(patterns_df.shape[0]) + ' rows and ' + str(patterns_df.shape[1]) + ' columns.')
        st.dataframe(patterns_df)
        st.markdown(filedownload(patterns_df), unsafe_allow_html=True)

    with col2:
        st.subheader('Top 20 Frequent Patterns (Bar Chart)')
        fig = plt.figure(figsize=(15, 10))
        # Remove the '%' symbol from 'Support' and 'Confidence' columns, and then convert to float
        top20_patterns_df['Support'] = patterns_df['Support'].str.rstrip('%').astype(float)
        top20_patterns_df['Confidence'] = patterns_df['Confidence'].str.rstrip('%').astype(float)
        try:
            sns.barplot(x="Support", y="Pattern", data=top20_patterns_df)
        except:
            pass
        else:
            st.pyplot(fig)

        st.subheader('Top 20 Frequent Patterns (Pie Chart)')
        fig1, ax1 = plt.subplots()
        ax1.pie(top20_patterns_df['Support'], labels=top20_patterns_df['Pattern'], colors=sns.color_palette('Set2'))
        ax1.axis('equal')
        st.pyplot(fig1)

def run_dashboard():
    algorithm_choices = {
        "Apriori": "Apriori Configuration",
        "FPGrowth": "FPGrowth Configuration",
        "GSP": "GSP Configuration",
        "PrefixSpan": "PrefixSpan Configuration",
    }

    algorithm_choice = st.sidebar.selectbox("Select algorithm type:", list(algorithm_choices.keys()))

    st.sidebar.header(algorithm_choices[algorithm_choice])
    with st.sidebar.form("user_form"):
        user_min_sup = st.slider("Min Support (%)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        user_min_confidence = st.slider("Min Confidence (%)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        user_min_pattern_length = st.slider("Min Pattern Length", min_value=1, max_value=len(name_list), value=1, step=1)
        user_excluded_features = st.multiselect("Excluded Features", sorted_name_list, [])
        generate_pattern = st.form_submit_button("Generate Pattern")

    if generate_pattern:
        if algorithm_choice == "Apriori":
            patterns_df = run_apriori(medical_condition_df,user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features)
            visualize_pattern_generation(patterns_df, patterns_df.head(20), algorithm_choice)

        elif algorithm_choice == "FPGrowth":
            patterns_df = run_fpgrowth(medical_condition_df,user_min_sup, user_min_confidence, user_min_pattern_length,user_excluded_features)
            visualize_pattern_generation(patterns_df, patterns_df.head(20), algorithm_choice)
    
        elif algorithm_choice == "GSP":
            patterns_df = run_gsp(patterns_gsp, user_min_sup, user_min_confidence, user_min_pattern_length,user_excluded_features)
            visualize_pattern_generation(patterns_df, patterns_df.head(20), algorithm_choice)

        elif algorithm_choice == "PrefixSpan":
            patterns_df = run_prefixspan(patterns_prefix, user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features, name_list)
            visualize_pattern_generation(patterns_df, patterns_df.head(20), algorithm_choice)

if __name__ == "__main__":
    medical_condition_df = pd.read_csv('Medical_Condition_Data.csv', index_col='Unnamed: 0')
    medical_condition_df = medical_condition_df.drop('DUTY_OF_DISCLOSURE', axis=1)
    medical_condition_df_subset = medical_condition_df.head(1600)
    
    #Preprocess data for GSP
    patterns_gsp =  []
    for _,row in medical_condition_df_subset.iloc[:,1:].iterrows():
        pattern = []
        for column_name, value in row.items():
            if value == 1:
                pattern.append(column_name)
        patterns_gsp.append(pattern)
    
    #Preprocess data for PrefixSpan 
    patterns_prefix = []
    for _, row in medical_condition_df.iloc[:, 1:].iterrows():
        pattern = []
        for i in range (0, len(row)):
            if row[i] == 1:
                pattern.append(i)
        patterns_prefix.append(pattern)

    # Get the name list:
    name_list = list(medical_condition_df.columns)
    sorted_name_list = sorted(name_list)

    run_dashboard()