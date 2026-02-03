import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO

try:
    import joblib
except Exception:
    joblib = None

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Academic Risk Prediction System",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Academic Risk Prediction System")
# center caption for better alignment
st.markdown(
    "<div style='text-align:center; color:gray;'>A decision-support system for early identification of at-risk students</div>",
    unsafe_allow_html=True,
)

# ----- layout / alignment CSS -----
st.markdown(
    """
    <style>
    .stApp h1, .stApp h2, .stApp h3 {text-align: center;}
    .reportview-container .markdown-text-container, .css-1d391kg {text-align: left;}
    .stButton>button, .stDownloadButton>button {width:100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "model.pkl"
SAMPLE_CSV = "student_data.csv"

def load_model(path=MODEL_PATH):
    if joblib is None:
        return None
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

model = load_model()

# ---------------- SIDEBAR: Data & Model ----------------
st.sidebar.header("Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload student CSV for batch predictions", type=["csv"])
use_sample = st.sidebar.button("Use sample dataset")

if model is not None:
    st.sidebar.success("Model loaded: model.pkl")
else:
    st.sidebar.warning("No trained model found. Batch predictions will use rule-based fallback.")

st.sidebar.markdown("---")
if st.sidebar.button("Run training (runs train_model.py)"):
    st.sidebar.info("Training started — run `python train_model.py` in terminal.")

st.markdown("---")

# ---------------- LAYOUT: Manual Input | Batch ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📥 Manual Student Inputs")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    marks = st.slider("Internal Marks (%)", 0, 100, 60)
    study_hours = st.slider("Average Study Hours / Day", 0, 10, 3)

    if st.button("🔍 Analyze Single Student"):
        # reuse rule-based logic as a fallback / explainable baseline
        risk_score = 0
        if attendance < 75:
            risk_score += 1
        if marks < 40:
            risk_score += 1
        if study_hours < 2:
            risk_score += 1

        probability = min(0.3 + risk_score * 0.2, 0.9)

        if risk_score >= 2:
            risk_label = "High Risk"
            color = "red"
        elif risk_score == 1:
            risk_label = "Moderate Risk"
            color = "orange"
        else:
            risk_label = "Low Risk"
            color = "green"

        st.subheader("📊 Risk Assessment Result")
        st.markdown(
            f"""**Academic Risk Level:**  
            <span style='color:{color}; font-size:22px'><b>{risk_label}</b></span>  
            **Confidence Score:** {int(probability*100)}%""",
            unsafe_allow_html=True,
        )

        # Simple bar chart
        fig1, ax1 = plt.subplots()
        ax1.bar(["Low", "Moderate", "High"], [1-probability, probability/2, probability])
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        st.subheader("🔎 Key Risk Factors Identified")
        if attendance < 75:
            st.write("• Low attendance impacting academic consistency")
        if marks < 40:
            st.write("• Weak internal performance detected")
        if study_hours < 2:
            st.write("• Insufficient daily study duration")
        if risk_score == 0:
            st.write("• No major academic risk indicators found")

        st.markdown("---")
        st.subheader("✅ Recommended Institutional Actions")
        if risk_label == "High Risk":
            st.write("• Assign academic mentor  • Weekly tracking  • Counseling intervention")
        elif risk_label == "Moderate Risk":
            st.write("• Monthly academic review  • Attendance monitoring")
        else:
            st.write("• Maintain current performance  • Encourage peer mentoring")

with col2:
    st.subheader("📂 Batch Predictions")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded CSV loaded")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    if use_sample or (uploaded_file is None and st.button("Load sample dataset")):
        if os.path.exists(SAMPLE_CSV):
            df = pd.read_csv(SAMPLE_CSV)
            st.info("Sample dataset loaded")
        else:
            st.error("Sample dataset not found in repo")

    if df is not None:
        st.write("Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # Predict function: use trained model if available, else fallback
        def predict_df(input_df):
            out = input_df.copy()
            if model is not None and hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(input_df.select_dtypes(include=[np.number]))
                    # assume positive class is last
                    out["risk_probability"] = probs[:, -1]
                except Exception:
                    out["risk_probability"] = 0.0
            else:
                # rule-based fallback using available numeric columns
                def rule_row(r):
                    a = r.get("attendance", r.get("Attendance", np.nan))
                    m = r.get("marks", r.get("Marks", np.nan))
                    s = r.get("study_hours", r.get("Study Hours", np.nan))
                    score = 0
                    if pd.notna(a) and a < 75:
                        score += 1
                    if pd.notna(m) and m < 40:
                        score += 1
                    if pd.notna(s) and s < 2:
                        score += 1
                    return min(0.3 + score * 0.2, 0.9)

                out["risk_probability"] = input_df.apply(rule_row, axis=1)

            out["risk_label"] = out["risk_probability"].apply(
                lambda p: "High Risk" if p >= 0.6 else ("Moderate Risk" if p >= 0.4 else "Low Risk")
            )
            return out

        if st.button("Run Batch Predictions"):
            with st.spinner("Running predictions..."):
                result_df = predict_df(df)
                st.success("Predictions complete")
                st.dataframe(result_df.head(20), use_container_width=True)

                # provide download
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                # show simple explanation for first row
                st.markdown("---")
                st.subheader("🔎 Explanation (first example)")
                example = result_df.iloc[0]
                st.write(example.to_dict())

                # Model-agnostic ablation explanations (replace each numeric feature with mean)
                if model is not None and hasattr(model, "predict_proba"):
                    try:
                        X_num = df.select_dtypes(include=[np.number])
                        if X_num.shape[1] == 0:
                            st.info("No numeric features available for ablation explanations.")
                        else:
                            X_num_filled = X_num.fillna(X_num.mean())
                            # baseline prediction for the example
                            row = X_num_filled.iloc[[0]]
                            try:
                                base_prob = model.predict_proba(row)[0, -1]
                            except Exception:
                                base_prob = float(model.predict(row)[0]) if hasattr(model, "predict") else 0.0

                            deltas = []
                            means = X_num_filled.mean()
                            for feat in X_num_filled.columns:
                                modified = row.copy()
                                modified[feat] = means[feat]
                                try:
                                    p = model.predict_proba(modified)[0, -1]
                                except Exception:
                                    p = float(model.predict(modified)[0]) if hasattr(model, "predict") else base_prob
                                deltas.append((feat, base_prob - p))

                            deltas_df = pd.DataFrame(deltas, columns=["feature", "delta"]).assign(abs_delta=lambda d: d["delta"].abs()).sort_values("abs_delta", ascending=False).head(10)
                            st.write("Top feature impacts (ablation - positive means current value increases risk):")
                            st.dataframe(deltas_df[["feature", "delta"]].set_index("feature"))

                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.barh(deltas_df["feature"], deltas_df["delta"])
                            ax.set_xlabel("Probability delta")
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Ablation explanation failed: {e}")
                elif model is not None and hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    try:
                        feature_names = model.feature_names_in_
                    except Exception:
                        feature_names = [f"f{i}" for i in range(len(fi))]
                    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
                    st.write("Top features (model)")
                    st.dataframe(fi_df.head(10))
                else:
                    st.info("No trained model available. To enable per-prediction explanations: place a trained `model.pkl` in the repo.")

    else:
        st.info("Upload a CSV or load the sample dataset to run batch predictions.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
    Maintainer: Himanshu Jadhav — Second-Year Engineering Student (AI & Data Science)  

    Connect: [GitHub](https://github.com/himanshu-jadhav108) • [LinkedIn](https://www.linkedin.com/in/himanshu-jadhav-328082339) • [Instagram](https://www.instagram.com/himanshu_jadhav_108) • [Portfolio](https://himanshu-jadhav-portfolio.vercel.app/)
    </div>
    """,
    unsafe_allow_html=True,
)
