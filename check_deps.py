
import streamlit as st

st.title("Dependency Check")

try:
    import dlib
    st.success(f"dlib installed: {dlib.__version__}")
except Exception as e:
    st.error(f"dlib error: {e}")

try:
    import face_recognition
    st.success("face_recognition installed successfully")
except Exception as e:
    st.error(f"face_recognition error: {e}")
