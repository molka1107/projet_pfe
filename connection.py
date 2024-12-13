import streamlit as st
import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  
            database='stage',  
            user='root',  
            password=''  
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Erreur lors de la connexion à MySQL: {e}")
    return None
