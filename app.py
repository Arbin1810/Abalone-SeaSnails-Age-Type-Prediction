#This dataset contains measurements of abalone (sea snails), used to predict their age (in rings).The rings are determined by cutting the shell, staining it, and counting the number of rings through a microscope — a time-consuming process. Type (M, F, I) — categorical (sex: Male, Female, Infant) 
# Rings — integer (target variable: age = rings + 1.5)

#reading the dataset
import pandas as pd
file_path = r'abalone.csv'
abalone_df = pd.read_csv(file_path)
# print(abalone_df.describe())
# print(abalone_df.head())
# print(abalone_df.tail())


#Cleaning the datasets

# print(abalone_df.dtypes)

# for column in abalone_df.columns:
#     unique_values = abalone_df[column].unique()
#     print(f"Column: {column}")
#     print(f"Unique values: {unique_values}")
    
#by analyzing the values the value in VisceraWeight seems to have a lot of decimals so i consider to round it in 4 decimal value for simplicity
abalone_df['VisceraWeight'] = abalone_df['VisceraWeight'].round(4)
#and i havenot found any duplicate or missing values or null value here 

#outliers checking using box plot
import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(figsize=(10, 6))
# abalone_df.boxplot(ax=ax)
# plt.title('Box Plot for Outlier Detection')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#From the plot i found a lot of outliers so now removing this to make my dataset more reliable for predictions
# Selecting numeric columns
numeric_cols = abalone_df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    Q1 = abalone_df[col].quantile(0.25)
    Q3 = abalone_df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
   
    abalone_df = abalone_df[(abalone_df[col] > lower_limit) & (abalone_df[col] < upper_limit)]

# print(abalone_df)
# print(abalone_df.dtypes)
# print(abalone_df.head())
#by removing the outliers from the data set total of 4177 now the value becamed 3773 

#rechecking the boxplot
# fig, ax = plt.subplots(figsize=(10, 6))
# abalone_df.boxplot(ax=ax)
# plt.title('Box Plot for Outlier Detection')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

#now my dataset is totally clean

#Visualization 
import streamlit as st
from test import train_and_compare_models
# Set page config for wider displaying
st.set_page_config(page_title="Abalone(SeaSnails) Age Prediction by Arbind", layout="wide")

# Display the data
st.title("Abalone(SeaSnails) Age Prediction Dashboard")
st.write(f"Dataset shape: {abalone_df.shape}")
st.dataframe(abalone_df)


st.header("Interactive Plots")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Interactive Barchart")
    st.subheader('Choose Your Parameter')
    
    parameter1 = st.selectbox(
        'Select abalone parameter:',
        ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
         'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'],
        key='barchart_param'
    )
    
    num_bars = st.slider('Number of bars:', 5, 50, 20, key='num_bars')
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))  
    bars = ax1.bar(range(num_bars), abalone_df[parameter1].head(num_bars))
    ax1.set_title(f'{parameter1} Values (First {num_bars} samples)')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel(parameter1)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.header("2. Interactive Boxplot")
    st.subheader('Choose Your Parameter')
    
    parameter2 = st.selectbox(
        'Select abalone parameter:',
        ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
         'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'],
        key='boxplot_param'
    )
    
    fig2, ax2 = plt.subplots(figsize=(5.7, 4)) 
    bp = ax2.boxplot(abalone_df[parameter2])
    mean_value = abalone_df[parameter2].mean()
    ax2.axhline(y=mean_value, color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {mean_value:.2f}')
    ax2.set_ylabel(parameter2)
    ax2.set_title(f'Distribution of {parameter2}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig2)

col1, col2 = st.columns(2)

with col1:
    st.subheader('3. Interactive Scatterplot')
    
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_axis = st.selectbox('X-axis:', 
                              ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
                               'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'],
                              key='scatter_x')
    with scatter_col2:
        y_axis = st.selectbox('Y-axis:', 
                              ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
                               'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'],
                              key='scatter_y')
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(abalone_df[x_axis], abalone_df[y_axis], 
                alpha=0.6, s=15, color='green')
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    ax1.set_title(f'{x_axis} vs {y_axis}')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('4. Interactive Histogram')
    
    parameter = st.selectbox(
        'Select abalone parameter:',
        ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
         'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'],
        key='hist_param'
    )
    
    num_bins = st.slider('Number of bins:', 5, 50, 20, key='hist_bins')
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    n, bins, patches = ax2.hist(abalone_df[parameter], 
                               bins=num_bins, 
                               edgecolor='black', 
                               alpha=0.7,
                               color='aqua')
    
    ax2.set_xlabel(parameter)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of {parameter}')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

st.header("Other Sample Plots")
st.header("Linecharts")

col1, col2 = st.columns(2)
with col1:
    st.write('1. Rings (Age) Analysis')
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(abalone_df.index[:100], abalone_df['Rings'].head(100), 
             label='Rings (Age)', color='blue', linewidth=2)
    ax1.axhline(y=abalone_df['Rings'].mean(), color='red', linestyle='--', 
               alpha=0.5, label=f'Mean ({abalone_df["Rings"].mean():.1f})')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Rings')
    ax1.set_title('Rings Variation', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.write('2. Rings vs Whole Weight (First 50)')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(abalone_df.index[:50], abalone_df['Rings'].head(50), 
             label='Rings', color='blue', linewidth=1.5)
    ax2.plot(abalone_df.index[:50], abalone_df['WholeWeight'].head(50), 
             label='Whole Weight', color='red', linewidth=1.5)
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Measurement')
    ax2.set_title('Rings vs Whole Weight', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader('Raw vs Moving Average (Rings)')
    window = st.slider('Smoothing window size:', 1, 20, 5, key='window_slider')
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(abalone_df.index[:100], abalone_df['Rings'].head(100), 
             label='Raw Rings', alpha=0.6, color='blue', linewidth=1)
    ax3.plot(abalone_df.index[:100], abalone_df['Rings'].head(100).rolling(window).mean(), 
             label=f'Moving Avg (window={window})', linewidth=2.5, color='red')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Rings')
    ax3.set_title('Rings: Raw vs Smoothed')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

with col4:
    st.subheader('Multiple Parameters Comparison')
    fig4, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    axes[0,0].plot(abalone_df.index[:100], abalone_df['LongestShell'].head(100), color='blue')
    axes[0,0].set_title('Longest Shell', fontsize=10)
    axes[0,0].grid(True, alpha=0.2)
    
    axes[0,1].plot(abalone_df.index[:100], abalone_df['WholeWeight'].head(100), color='orange')
    axes[0,1].set_title('Whole Weight', fontsize=10)
    axes[0,1].grid(True, alpha=0.2)
    
    axes[1,0].plot(abalone_df.index[:100], abalone_df['Diameter'].head(100), color='green')
    axes[1,0].set_title('Diameter', fontsize=10)
    axes[1,0].grid(True, alpha=0.2)
    
    axes[1,1].plot(abalone_df.index[:100], abalone_df['ShuckedWeight'].head(100), color='purple')
    axes[1,1].set_title('Shucked Weight', fontsize=10)
    axes[1,1].grid(True, alpha=0.2)
    
    plt.suptitle('Four Key Abalone Parameters', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig4)

# Barcharts
st.header("Barcharts")
col1, col2 = st.columns(2)

with col1:
    st.subheader('1. Average Values of Parameters')
    avg_values = {
        'LongestShell': abalone_df['LongestShell'].mean(),
        'Diameter': abalone_df['Diameter'].mean(),
        'WholeWeight': abalone_df['WholeWeight'].mean(),
        'Rings': abalone_df['Rings'].mean()
    }
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(avg_values.keys(), avg_values.values(), 
                   color=['blue', 'orange', 'green', 'purple'])
    ax1.set_ylabel('Average Value')
    ax1.set_title('Average Abalone Parameters')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('2. Average Values by Sex')
    male_avg = abalone_df[abalone_df['Type'] == 'M'][['LongestShell', 'Rings']].mean()
    female_avg = abalone_df[abalone_df['Type'] == 'F'][['LongestShell', 'Rings']].mean()
    infant_avg = abalone_df[abalone_df['Type'] == 'I'][['LongestShell', 'Rings']].mean()
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.25
    
    bars1 = ax2.bar(x - width, male_avg.values, width, 
                   label='Male (M)', color='blue', edgecolor='black')
    bars2 = ax2.bar(x, female_avg.values, width, 
                   label='Female (F)', color='red', edgecolor='black')
    bars3 = ax2.bar(x + width, infant_avg.values, width, 
                   label='Infant (I)', color='green', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Longest Shell', 'Rings'])
    ax2.set_ylabel('Average Value')
    ax2.set_title('Comparison by Sex')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

# Boxplots
st.header("Boxplots")
col1, col2 = st.columns(2)
with col1:
    st.subheader('1. Rings Distribution')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.boxplot(abalone_df['Rings'])
    ax1.set_ylabel('Rings (Age)')
    ax1.set_title('Overall Rings Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig1)

with col2:
    st.subheader('2. Rings by Sex')
    male_rings = abalone_df[abalone_df['Type'] == 'M']['Rings']
    female_rings = abalone_df[abalone_df['Type'] == 'F']['Rings']
    infant_rings = abalone_df[abalone_df['Type'] == 'I']['Rings']
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    box = ax2.boxplot([male_rings, female_rings, infant_rings], 
                     tick_labels=['Male', 'Female', 'Infant'],
                     patch_artist=True)
    
    colors = ['lightblue', 'pink', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Rings (Age)')
    ax2.set_title('Rings by Sex')
    ax2.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig2)

# Piecharts
st.header("Piecharts")
col1, col2 = st.columns(2)

with col1:
    st.subheader('1. Sex Distribution')
    male_count = len(abalone_df[abalone_df['Type'] == 'M'])
    female_count = len(abalone_df[abalone_df['Type'] == 'F'])
    infant_count = len(abalone_df[abalone_df['Type'] == 'I'])
    
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax1.pie(
        [male_count, female_count, infant_count], 
        labels=['Male (M)', 'Female (F)', 'Infant (I)'],
        autopct='%1.1f%%',
        colors=['blue', 'red', 'green'],
    )
    ax1.set_title('Abalone Sex Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('2. Age Categories by Rings')
    
    def categorize_rings(rings):
        if rings <= 5:
            return 'Young (≤5 rings)'
        elif 6 <= rings <= 10:
            return 'Middle (6-10 rings)'
        else:
            return 'Old (>10 rings)'
    
    abalone_df['Ring_category'] = abalone_df['Rings'].apply(categorize_rings)
    category_counts = abalone_df['Ring_category'].value_counts()
    
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(category_counts.values, 
           labels=category_counts.index,
           autopct='%1.1f%%',
           colors=['lightgreen', 'orange', 'brown'],
          )
    ax2.set_title('Age Categories by Rings', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig2)

# Scatterplots
st.header("Scatter plots")
col1, col2 = st.columns(2)

with col1:
    st.subheader('Whole Weight vs Rings')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(abalone_df['WholeWeight'], abalone_df['Rings'], 
               alpha=0.6, color='blue', s=20)
    ax1.set_xlabel('Whole Weight')
    ax1.set_ylabel('Rings (Age)')
    ax1.set_title('Whole Weight vs Rings')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('Longest Shell vs Diameter by Sex')
    
    male = abalone_df[abalone_df['Type'] == 'M']
    female = abalone_df[abalone_df['Type'] == 'F']
    infant = abalone_df[abalone_df['Type'] == 'I']
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(male['LongestShell'], male['Diameter'], 
               color='blue', alpha=0.6, label='Male', s=20)
    ax2.scatter(female['LongestShell'], female['Diameter'], 
               color='red', alpha=0.6, label='Female', s=20)
    ax2.scatter(infant['LongestShell'], infant['Diameter'], 
               color='green', alpha=0.6, label='Infant', s=20)
    
    ax2.set_xlabel('Longest Shell')
    ax2.set_ylabel('Diameter')
    ax2.set_title('Shell Dimensions by Sex')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)




#4. Train the different ML model for you problem and show barchart for their accuracy metrics [KNN, SVR, Logistic Regrssion] (best model after hyperparameter tuning) (In Streamlit Dashboard)
# for regression we gonna analyze among : KNN (REGRESSION), SVR, linear regression

# ML Model Training Section
st.header("4. Machine Learning Models Comparison")

# Remove Ring_category column before training
if 'Ring_category' in abalone_df.columns:
    abalone_df = abalone_df.drop(columns=['Ring_category'])

# Train models and get metrics
with st.spinner("Training models and tuning hyperparameters..."):
    metrics_data, scaler, feature_names = train_and_compare_models(abalone_df)

# Display best parameters
st.subheader("Best Hyperparameters Found:")
for model_name, data in metrics_data.items():
    st.write(f"**{model_name}:** {data['best_params']}")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(metrics_data.keys()),
    'MSE': [metrics_data[m]['mse'] for m in metrics_data.keys()],
    'R² Score': [metrics_data[m]['r2'] for m in metrics_data.keys()],
    'MAE': [metrics_data[m]['mae'] for m in metrics_data.keys()],
    'RMSE': [metrics_data[m]['rmse'] for m in metrics_data.keys()]
})

st.subheader("Model Performance Metrics (After Hyperparameter Tuning)")
st.dataframe(comparison_df)

# Find best model
best_model_name = max(metrics_data.keys(), key=lambda x: metrics_data[x]['r2'])
best_model = metrics_data[best_model_name]['model']

st.subheader(f"Best Model: {best_model_name}")
st.write(f"**R² Score:** {metrics_data[best_model_name]['r2']:.3f}")
st.write(f"**RMSE:** {metrics_data[best_model_name]['rmse']:.3f}")

# PREDICTION INTERFACE FOR ABALONE AGE
st.header("Predict Abalone Age")

age_category_mapping = {
    'Young (≤5 rings)': "Young Abalone (Age: ≤5 rings)",
    'Middle (6-10 rings)': "Middle-aged Abalone (Age: 6-10 rings)",
    'Old (>10 rings)': "Old Abalone (Age: >10 rings)"
}

st.subheader("Enter Abalone Measurements:")

col1, col2 = st.columns(2)

with col1:
    longest_shell = st.number_input("Longest Shell (mm)", min_value=0.0, max_value=1.0, 
                                    value=0.5, step=0.01, help="Length of the longest shell measurement")
    diameter = st.number_input("Diameter (mm)", min_value=0.0, max_value=1.0, 
                               value=0.4, step=0.01, help="Diameter measurement")
    height = st.number_input("Height (mm)", min_value=0.0, max_value=1.0, 
                             value=0.15, step=0.01, help="Height measurement")
    whole_weight = st.number_input("Whole Weight (grams)", min_value=0.0, max_value=3.0, 
                                   value=0.8, step=0.01, help="Total weight of the abalone")

with col2:
    shucked_weight = st.number_input("Shucked Weight (grams)", min_value=0.0, max_value=2.0, 
                                     value=0.3, step=0.01, help="Weight of meat after shucking")
    viscera_weight = st.number_input("Viscera Weight (grams)", min_value=0.0, max_value=1.0, 
                                     value=0.15, step=0.01, help="Weight of gut organs")
    shell_weight = st.number_input("Shell Weight (grams)", min_value=0.0, max_value=1.5, 
                                   value=0.2, step=0.01, help="Weight of the shell")
    
    # Type selection
    type = st.selectbox("Types", ["Male (M)", "Female (F)", "Infant (I)"], 
                       help="Select the type of the abalone")

# Create one-hot encoding for sex
if type == "Male (M)":
    type_f = 0
    type_i = 0
elif type == "Female (F)":
    type_f = 1
    type_i = 0
else:  # Infant (I)
    type_f = 0
    type_i = 1

# Create user input array in the same order as training
user_data = [[longest_shell, diameter, height, whole_weight, 
              shucked_weight, viscera_weight, shell_weight, type_f, type_i]]

# Prediction button
if st.button("Predict Abalone Age"):
    # Scale the input data using the same scaler
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction using best model
    predicted_rings = best_model.predict(user_data_scaled)[0]
    
    # Calculate actual age (age = rings + 1.5 years)
    actual_age = predicted_rings + 1.5
    
    # Categorize age
    if predicted_rings <= 5:
        age_category = 'Young (≤5 rings)'
    elif 6 <= predicted_rings <= 10:
        age_category = 'Middle (6-10 rings)'
    else:
        age_category = 'Old (>10 rings)'
    
    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # result display
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Predicted Rings", f"{predicted_rings:.1f}")
    
    with result_col2:
        st.metric("Estimated Age", f"{actual_age:.1f} years")
    
    with result_col3:
        if age_category == 'Young (≤5 rings)':
            st.success(age_category_mapping[age_category])
        elif age_category == 'Middle (6-10 rings)':
            st.warning(age_category_mapping[age_category])
        else:
            st.error(age_category_mapping[age_category])