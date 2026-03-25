import streamlit as st
import pandas as pd
import joblib
import os
import hashlib

st.set_page_config(page_title="Job Predictor Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .css-1d391kg {
        background-color: #2c3e50;
    }
    .css-1v3fvcr {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return None

def load_model_and_encoders():
    try:
        model = joblib.load("job_model.pkl")
        degree_enc = joblib.load("degree_encoder.pkl")
        spec_enc = joblib.load("spec_encoder.pkl")
        job_enc = joblib.load("job_encoder.pkl")
        return model, degree_enc, spec_enc, job_enc
    except FileNotFoundError:
        return None, None, None, None

def load_dataset():
    if os.path.exists("job_dataset.csv"):
        return pd.read_csv("job_dataset.csv")
    return None

def init_user_db():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["username", "password", "name", "degree", "specialization", "cgpa"])
        df.to_csv("users.csv", index=False)

def register_user(username, password, name):
    init_user_db()
    users = pd.read_csv("users.csv")
    if username in users['username'].values:
        return False
    new_user = pd.DataFrame({
        "username": [username],
        "password": [make_hashes(password)],
        "name": [name],
        "degree": [""],
        "specialization": [""],
        "cgpa": [0.0]
    })
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)
    return True

def login_user(username, password):
    init_user_db()
    users = pd.read_csv("users.csv")
    user = users[users['username'] == username]
    if not user.empty:
        if check_hashes(password, user['password'].values[0]):
            return user
    return None

def update_profile(username, name, degree, spec, cgpa):
    users = pd.read_csv("users.csv")
    users.loc[users['username'] == username, 'name'] = name
    users.loc[users['username'] == username, 'degree'] = degree
    users.loc[users['username'] == username, 'specialization'] = spec
    users.loc[users['username'] == username, 'cgpa'] = cgpa
    users.to_csv("users.csv", index=False)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

    model, degree_enc, spec_enc, job_enc = load_model_and_encoders()
    data = load_dataset()

    st.sidebar.title("Navigation")
    
    if st.session_state.logged_in:
        menu = ["Dashboard", "Job Prediction", "My Profile", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
    else:
        menu = ["Home", "Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("🎓 Career Path Predictor")
        st.write("Welcome to the Job Role Prediction System.")
        st.info("Please Login or Register to access the dashboard and prediction tools.")
        if data is not None:
            st.subheader("Dataset Overview")
            st.write(data.head())

    elif choice == "Login":
        st.title("🔐 User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                result = login_user(username, password)
                if result is not None:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_data = result
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

    elif choice == "Register":
        st.title("📝 New User Registration")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        new_name = st.text_input("Full Name")
        
        if st.button("Register"):
            if not new_user or not new_password or not new_name:
                st.error("All fields are required.")
            else:
                if register_user(new_user, new_password, new_name):
                    st.success("Account created successfully! Please Login.")
                else:
                    st.error("Username already exists.")

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_data = None
        st.success("Logged out successfully.")
        st.rerun()

    elif choice == "Dashboard":
        st.title(f"👋 Welcome, {st.session_state.user_data['name'].values[0]}!")
        st.write("This is your personal dashboard. Use the sidebar to navigate.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Active")
        with col2:
            st.metric("Profile Completion", "50%" if st.session_state.user_data['degree'].values[0] else "0%")
        with col3:
            st.metric("Predictions Made", "0")

        st.subheader("Quick Actions")
        if st.button("Go to Prediction"):
            st.session_state.predict_focus = True
            st.rerun()

    elif choice == "My Profile":
        st.title("👤 User Profile")
        
        users = pd.read_csv("users.csv")
        current_user = users[users['username'] == st.session_state.username].iloc[0]
        
        with st.form("profile_form"):
            name = st.text_input("Full Name", value=current_user['name'])
            
            if degree_enc:
                degree_opts = degree_enc.classes_
                degree = st.selectbox("Degree", degree_opts, index=0 if current_user['degree'] == "" else list(degree_opts).index(current_user['degree']) if current_user['degree'] in degree_opts else 0)
                spec_opts = spec_enc.classes_
                spec = st.selectbox("Specialization", spec_opts, index=0 if current_user['specialization'] == "" else list(spec_opts).index(current_user['specialization']) if current_user['specialization'] in spec_opts else 0)
            else:
                degree = st.text_input("Degree", value=current_user['degree'])
                spec = st.text_input("Specialization", value=current_user['specialization'])
            
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=float(current_user['cgpa']) if current_user['cgpa'] else 0.0)
            
            submitted = st.form_submit_button("Save Profile")
            
            if submitted:
                if not name or not degree or not spec:
                    st.error("Name, Degree, and Specialization cannot be empty.")
                elif cgpa < 0 or cgpa > 10:
                    st.error("CGPA must be between 0 and 10.")
                else:
                    update_profile(st.session_state.username, name, degree, spec, cgpa)
                    st.success("Profile Updated Successfully!")
                    st.rerun()

    elif choice == "Job Prediction":
        st.title("🔮 Job Role Prediction")
        
        if model is None:
            st.error("Model files not found! Please run train_model.py first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.form("prediction_form"):
                    st.subheader("Enter Academic Details")
                    deg_input = st.selectbox("Degree", degree_enc.classes_)
                    spec_input = st.selectbox("Specialization", spec_enc.classes_)
                    cgpa_input = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
                    
                    predict_btn = st.form_submit_button("Predict Job Role")
                    
                    if predict_btn:
                        if cgpa_input < 0 or cgpa_input > 10:
                            st.error("CGPA must be between 0 and 10.")
                        else:
                            try:
                                deg_encoded = degree_enc.transform([deg_input])[0]
                                spec_encoded = spec_enc.transform([spec_input])[0]
                                
                                input_data = pd.DataFrame([[deg_encoded, spec_encoded, cgpa_input]], 
                                                          columns=["Degree", "Specialization", "CGPA"])
                                prediction = model.predict(input_data)
                                
                                job_role = job_enc.inverse_transform(prediction)[0]
                                
                                st.success(f"Predicted Job Role: **{job_role}**")
                            except Exception as e:
                                st.error(f"Prediction Error: {e}")

            with col2:
                st.subheader("Job Distribution")
                if data is not None:
                    job_counts = data['JobRole'].value_counts()
                    st.bar_chart(job_counts)
                else:
                    st.warning("Dataset not found for visualization.")

if __name__ == '__main__':
    main()