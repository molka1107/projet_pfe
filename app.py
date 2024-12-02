import streamlit as st
from connection import create_connection
from mysql.connector import Error
import subprocess 
import time 


st.set_page_config(
    page_title="Assistant Virtuel pour Personnes Malvoyantes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .title-with-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: row-reverse;
        margin: 0 auto; 
    }
    .title-with-icon img {
        width: 70px;
        height: 70px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/assistant-virtuel-7660458-6297102.png?f=webp&w=256" alt="icone">
        <h1>Assistant virtuel pour les personnes malvoyantes</h1>
    </div>
""", unsafe_allow_html=True)


# Charger le CSS à partir d'un fichier externe
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "login"  


# Function to navigate between pages
def navigate_to(page_name):
    st.session_state["page"] = page_name  


# Function to open main.py on a separate port
def open_main_app():
    subprocess.Popen(["streamlit", "run", "main.py", "--server.port", "8504"])  


# Function to handle user login
def login_user(connection, email, password):
    try:
        cursor = connection.cursor()
        select_query = """
            SELECT * FROM users WHERE email=%s AND password=%s
        """
        cursor.execute(select_query, (email, password))
        user = cursor.fetchone()
        return user
    except Error as e:
        st.error(f"Échec de la récupération de l'enregistrement depuis la table MySQL: {e}")
    finally:
        cursor.close()
    return None


# Function to handle user signup
def signup_user(connection, name, username, email, password):
    try:
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO users (name, username, email, password)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, username, email, password))
        connection.commit()
        message_placeholder = st.empty()
        message_placeholder.success("Utilisateur enregistré avec succès !")
        time.sleep(3)
        message_placeholder.empty()
    except Error as e:
        st.error(f"Échec de la récupération de l'enregistrement depuis la table MySQL: {e}")
    finally:
        cursor.close()


# Login Page
def login_page():
    st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn-icons-png.flaticon.com/128/15227/15227593.png" alt="icone">
        <h2>Connexion</h2>
    </div>
    """, unsafe_allow_html=True)
    # Initialize session state for email and password
    if "login_email" not in st.session_state:
        st.session_state.login_email = ""
    if "login_password" not in st.session_state:
        st.session_state.login_password = ""


    # Input fields with session state binding
    email = st.text_input("E-mail", value=st.session_state.login_email, key="email_input")
    password = st.text_input("Mot de passe", type="password", value=st.session_state.login_password, key="password_input")
    # Log In button
    login_button = st.button("Se connecter")


    if login_button:
        if email and password:
            connection = create_connection()
            if connection:
                user = login_user(connection, email, password)
                connection.close()
                if user:
                    st.success("Connexion réussie !")
                    st.session_state['is_authenticated'] = True
                    st.session_state.login_email = ""  
                    st.session_state.login_password = ""  
                    open_main_app()  
                    st.stop()  
                else:
                    st.error("E-mail ou mot de passe incorrect :face_with_raised_eyebrow: Veuillez réessayer !")
        else:
            st.warning("Veuillez remplir tous les champs !")
    # Navigate to the signup page
    st.button("Vous n'avez pas de compte ? Créez-en un !", on_click=lambda: navigate_to("signup"))


# Signup Page
def signup_page():
    st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn-icons-png.flaticon.com/128/7981/7981789.png" alt="icone">
        <h2>Inscription</h2>
    </div>
    """, unsafe_allow_html=True)
    # Track signup success state in session state
    if "signup_success" not in st.session_state:
        st.session_state.signup_success = False
    # Signup form inputs
    name = st.text_input("Nom et prénom")
    username = st.text_input("Nom d'utilisateur")
    email = st.text_input("E-mail")
    password = st.text_input("Mot de passe", type='password')


    # Show "Sign Up" button only if signup is not yet successful
    if not st.session_state.signup_success:
        signup_button = st.button("S'inscrire")
        if signup_button:
            if name and username and email and password:
                connection = create_connection()
                if connection:
                    signup_user(connection, name, username, email, password)
                    connection.close()
                    st.session_state.signup_success = True 
                    message_placeholder = st.empty()
                    message_placeholder.success("Inscription réussie ! Veuillez vous connecter !")
                    time.sleep(3)
                    message_placeholder.empty()
            else:
                st.warning("Veuillez remplir tous les champs !")


    # After successful signup, show "Log In" button
    if st.session_state.signup_success:
        st.button("Se connecter", on_click=lambda: navigate_to("login"))
    # Add a secondary button for users who already have an account
    if not st.session_state.signup_success:
        st.button("Vous avez déjà un compte ? Connectez-vous !", on_click=lambda: navigate_to("login"))


# Route to the appropriate page
if st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "signup":
    signup_page()
