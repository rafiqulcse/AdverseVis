import base64
import pyautogui
import streamlit as st
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title('AdverseVis: Visual Interactive System for Adverse Behaviour Identification')

# State shape
states_shp = gpd.read_file("Pages/STE_2021_AUST_GDA2020/STE_2021_AUST_GDA2020.shp")
states_shp = states_shp.head(8)
states_shp['abbreviated_state_name'] = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
states_shp['number_of_people'] = 0

# Demographic DF
demographic_df = pd.read_excel('Pages/Demographic_Data.xlsx')


# Drop null rows
demographic_df = demographic_df.dropna(how='any')
demographic_df = demographic_df[~demographic_df['occupation_category'].isin(['HM ', 'W '])]

#Occupation name mapping
occupation_mapping = {
    "R ": "R - Special Risk",
    "E ": "E - Executive Income",
    "I - Indoor Sedentary": "I - Indoor Sedentary",
    "F ": "F - Finanical Professional",
    "D ": "D - Medical/Dental",
    "M ": "M - Mobile Professional",
    "OR ": "OR - Ordinary Rates",
    "L - Light Trades":"L - Light Trades",
    "P - Qualified Professional": "P - Qualified Professional",
    "C - Community Professional": "C - Community Professional",
    "H ": "H - Heavy Trades",
    "T - Trades": "T - Trades",
    "U ": "U - Uncategorised",
    "HH - Heavy Duties":"HH - Heavy Duties",
    "S ": "S - Supervisor of Trades",
    "A ": "A - Legal",
    "IC ": "IC - Individual Consideration",
    " " : "N - No Job"
}
demographic_df['occupation_category'] = demographic_df['occupation_category'].map(occupation_mapping)
unique_occupations = sorted(demographic_df['occupation_category'].astype(str).unique(), key=lambda x: x[0], reverse=True)

# Get state name
def get_state_name(post_code):
    post_code = int(post_code)
    state = None

    if (1000 <= post_code <= 2599) or (2619 <= post_code <= 2899) or (2921 <= post_code <= 2999):
        state = 'NSW'
    elif (200 <= post_code <= 299) or (2600 <= post_code <= 2618) or (2900 <= post_code <= 2920):
        state = 'ACT'
    elif (3000 <= post_code <= 3999) or (8000 <= post_code <= 8999):
        state = 'VIC'
    elif (4000 <= post_code <= 4999) or (9000 <= post_code <= 9999):
        state = 'QLD'
    elif (5000 <= post_code <= 5799) or (5800 <= post_code <= 5999):
        state = 'SA'
    elif (6000 <= post_code <= 6797) or (6800 <= post_code <= 6999):
        state = 'WA'
    elif (7000 <= post_code <= 7799) or (7800 <= post_code <= 7999):
        state = 'TAS'
    elif (800 <= post_code <= 899) or (900 <= post_code <= 999):
        state = 'NT'
    return state

demographic_df['abbreviated_state_name'] = demographic_df['life_insured_post_code'].apply(get_state_name)

# VISULIZATION:
st.sidebar.header('User Input')

with st.sidebar.form("my_form"):
    min_age = st.slider("Min Age", min_value=1, max_value=120, value=30, step=1)
    included_states = st.multiselect("Included States", sorted(states_shp['abbreviated_state_name'], key=lambda x: x[0]), [])
    submitted = st.form_submit_button("Identify Adverse Behaviour")

if submitted:
    # Check if user specify state:
    if len(included_states) == 0:
       included_states = states_shp['abbreviated_state_name']
    
    # Remove unincluded rows in the demographic_df:
    result_demographic_df = demographic_df[(demographic_df['abbreviated_state_name'].isin(included_states)) & (demographic_df['Age'] >= min_age)]

    # Update people count by state:
    count_of_value = result_demographic_df['abbreviated_state_name'].value_counts()
    result_states_shp = states_shp

    for index, count in count_of_value.items():
        result_states_shp.loc[result_states_shp['abbreviated_state_name'] == index, 'number_of_people'] = count

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Policy Holders by State")

        fig, ax = plt.subplots(figsize=(15, 10))

        excluded_states_shp = result_states_shp[~result_states_shp['abbreviated_state_name'].isin(included_states)]
        included_states_shp = result_states_shp[result_states_shp['abbreviated_state_name'].isin(included_states)]

        try:
            excluded_states_shp.plot(ax=ax, legend=False, color='lightgray', linewidth=0.5, edgecolor='0.8')
        except:
            pass

        try:
            included_states_shp.plot(ax=ax, legend=False, cmap='tab20', linewidth=0.5, edgecolor='0.8')
        except:
            pass

        ax.set_axis_off()

        for idx, row in included_states_shp.iterrows():
            ax.text(row['geometry'].centroid.x, row['geometry'].centroid.y, f"{row['abbreviated_state_name']}\n{row['number_of_people']}", ha='center', va='center', fontweight='bold', fontsize=9, color='white')

        st.pyplot(fig)
    with col2:
        st.subheader("Policy Holders by Occupation and Gender")
        # Set the style
        sns.set(style="whitegrid")

        # Create the plot
        fig = plt.figure(figsize=(15, 10))

        try:
            ax = sns.countplot(y='occupation_category', hue='life_insured_gender', data=result_demographic_df , orient='h', hue_order=['F ', 'M '], order=unique_occupations)
        except:
            pass

        # Adding improved annotations
        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                try:
                    plt.text(width + 10, p.get_y() + p.get_height() / 2, f'{int(width)}', va='center', fontsize=10)
                except:
                    pass

        # Customizing the plot
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Occupation', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Gender', fontsize=12, title_fontsize=12)
        plt.tight_layout()

        # Show the plot
        st.pyplot(fig)

        st.subheader("Policy Holders by Age Group")
        # Define age group bins and label
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        age_labels = ['1-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109', '110-119']

        # Assign age group labels to the dataset
        result_demographic_df ['age_group'] = pd.cut(result_demographic_df ['Age'], bins=age_bins, labels=age_labels, right=False)

        # Set the style
        sns.set(style="whitegrid")

        # Create the plot
        figure = plt.figure(figsize=(15, 10))

        try:
            ax = sns.countplot(x='age_group', hue='life_insured_gender', data=result_demographic_df , hue_order=['F ', 'M '])
        except:
            pass

        # Adding improved annotations
        for p in ax.patches:
            height = p.get_height()
            try:
                plt.text(p.get_x() + p.get_width() / 2, height + 5, f'{int(height)}', ha='center', fontsize=10)
            except:
                pass

        # Customizing the plot
        plt.xlabel('Age Group', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.legend(title='Gender', fontsize=12, title_fontsize=12, loc = "upper right")
        plt.tight_layout()

        # Show the plot
        st.pyplot(figure)

    if st.button("Reset"):
        pyautogui.hotkey("ctrl", "F5")
