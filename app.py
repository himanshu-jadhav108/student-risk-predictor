import streamlit as st

st.set_page_config(page_title="Student Risk Predictor")

st.title("AI-Based Student Risk Prediction System")

st.write("Enter student academic details to assess risk level.")

attendance = st.number_input("Attendance Percentage", 0, 100)
marks = st.number_input("Internal Marks", 0, 100)

if st.button("Predict Risk"):
    if attendance < 75 or marks < 40:
        st.error("High Risk Student")
        st.write("### Recommended Actions:")
        st.write("- Assign a faculty mentor")
        st.write("- Provide remedial classes")
        st.write("- Monitor weekly progress")
    else:
        st.success("Low Risk Student")
        st.write("### Recommended Actions:")
        st.write("- Continue current academic plan")
        st.write("- Periodic performance review")
