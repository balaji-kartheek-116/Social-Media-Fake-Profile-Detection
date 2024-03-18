import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to load data
def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, test_df

# Function to authenticate user
def authenticate(username, password):
    return username == "admin" and password == "password"

# Title
st.title('Social Media Fake Profile Detection')
st.image("profile.png")

# Session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login form
if not st.session_state.authenticated:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
        else:
            st.error("Invalid username or password")

# Main content
if st.session_state.authenticated:
    # Load data
    train_df, test_df = load_data()

    # Sample DataFrame
    st.subheader('Sample DataFrame')
    st.write(train_df.head())

    # Data Visualizations
    st.subheader('Data Visualizations')

    # Histogram of Description Length
    plt.figure(figsize=(12, 8))
    sns.histplot(train_df['description length'], bins=30, kde=True)
    plt.title('Distribution of Description Length')
    plt.xlabel('Description Length')
    plt.ylabel('Frequency')
    st.pyplot()
    st.markdown('---')

    # Countplot of Profile Picture
    plt.figure(figsize=(10, 6))
    sns.countplot(x='profile pic', data=train_df)
    plt.title('Profile Picture Distribution')
    plt.xlabel('Profile Picture')
    plt.ylabel('Count')
    st.pyplot()
    st.markdown('---')

    # Boxplot of Number of Posts by Fake/Real Profiles
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='fake', y='#posts', data=train_df)
    plt.title('Number of Posts by Fake/Real Profiles')
    plt.xlabel('Fake')
    plt.ylabel('Number of Posts')
    st.pyplot()
    st.markdown('---')

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot()
    st.markdown('---')

    # Barplot of Average Number of Followers by Fake/Real Profiles
    plt.figure(figsize=(12, 8))
    sns.barplot(x='fake', y='#followers', data=train_df)
    plt.title('Average Number of Followers by Fake/Real Profiles')
    plt.xlabel('Fake')
    plt.ylabel('Average Number of Followers')
    st.pyplot()
    st.markdown('---')

    # Model Training and Evaluation
    st.subheader('Model Training and Evaluation')

    # Data Preprocessing
    X_train = train_df.drop(columns=['fake'])
    y_train = train_df['fake']
    X_test = test_df.drop(columns=['fake'])
    y_test = test_df['fake']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='linear'),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    model_accuracies = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
        
        joblib.dump(model, f"{name}_model.pkl")

    # Display model accuracies
    st.subheader('Model Accuracies')
    model_accuracies_df = pd.DataFrame(model_accuracies.items(), columns=['Model', 'Accuracy'])
    st.write(model_accuracies_df)

    # Plot model accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
    plt.title('Model Accuracies')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    st.pyplot()
    st.markdown('---')

    # Prediction
    st.subheader('Fake Profile Detection')

    # Input fields
    profile_pic = st.selectbox('Profile Picture (0: No, 1: Yes)', [0, 1])
    nums_length_username = st.number_input('Numbers/Length of Username', min_value=0)
    fullname_words = st.number_input('Fullname Words', min_value=0)
    nums_length_fullname = st.number_input('Numbers/Length of Fullname', min_value=0)
    name_eq_username = st.selectbox('Name equals Username (0: No, 1: Yes)', [0, 1])
    description_length = st.number_input('Description Length', min_value=0)
    external_URL = st.selectbox('External URL (0: No, 1: Yes)', [0, 1])
    private = st.selectbox('Private (0: No, 1: Yes)', [0, 1])
    num_posts = st.number_input('Number of Posts', min_value=0)
    num_followers = st.number_input('Number of Followers', min_value=0)
    num_follows = st.number_input('Number of Follows', min_value=0)

    # Predict button
    if st.button('Predict'):
        # Prepare input data
        input_data = np.array([[profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_eq_username,
                                 description_length, external_URL, private, num_posts, num_followers, num_follows]])
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        # Select the best performing model (Random Forest in this case)
        model = joblib.load('Random Forest_model.pkl')
        # Predict
        prediction = model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.error('This profile is predicted to be fake.')
        else:
            st.success('This profile is predicted to be real.')

    # Logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
