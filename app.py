import streamlit as st

st.title("My First Streamlit App 🎉")
st.write("Hello, Streamlit!")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")
