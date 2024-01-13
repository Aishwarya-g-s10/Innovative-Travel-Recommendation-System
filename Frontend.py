import streamlit as st
import pandas as pd
import hashlib
import base64

# Function to hash passwords (not the most secure way, but for demonstration purposes)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if the user exists and the password is correct
def check_user(username, password):
    # Replace with your actual database or users file
    try:
        users = pd.read_csv('users.csv')
        user_info = users[users['username'] == username].iloc[0]
        return hash_password(password) == user_info['password']
    except:
        return False

# Function to add a new user (this is not secure and should only be for demonstration)
def add_user(username, password):
    # Replace with your actual database or users file
    users = pd.read_csv('users.csv')
    if username in users['username'].values:
        return False  # User already exists
    else:
        # Add new user
        users = users.append({'username': username, 'password': hash_password(password)}, ignore_index=True)
        users.to_csv('users.csv', index=False)
        return True

# Login/sign-up function with real implementation
def login_system():
    st.sidebar.title("Login/Signup")
    menu = ["Login", "Signup"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if check_user(username, password):
                st.session_state['user'] = username
                st.success(f"Logged in as {username}")
                return True
            else:
                st.error("Incorrect username or password.")
                return False

    elif choice == "Signup":
        new_username = st.sidebar.text_input("Choose a Username", key="new_user")
        new_password = st.sidebar.text_input("Choose a Password", type="password", key="new_pass")

        if st.sidebar.button("Signup"):
            if add_user(new_username, new_password):
                st.success("You have successfully signed up!")
                st.session_state['user'] = new_username
                return True
            else:
                st.error("Username is already taken.")
                return False

    return False

# Function to handle photo upload (returns the uploaded file)
def upload_photo():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Process the file (not implemented)
        return uploaded_file
    return None

# Function to display recommendations (dummy implementation)
def show_recommendations(uploaded_file):
    st.write("Showing recommendations for the uploaded photo (not actually implemented).")

# Function to add a background image from a local file
def add_bg_from_local():
    with open('background.jpg', "rb") as file:
        base64_image = base64.b64encode(file.read()).decode('utf-8')

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add a background image
add_bg_from_local()

# Main app logic
st.header("Travel Recommendation System")

if 'user' not in st.session_state:
    st.session_state['user'] = None

if login_system():  # If user is logged in
    uploaded_file = upload_photo()
    if uploaded_file is not None:  # If a file has been uploaded
        show_recommendations(uploaded_file)
elif 'user' in st.session_state and st.session_state['user']:
    st.write(f"You are logged in as {st.session_state['user']}")