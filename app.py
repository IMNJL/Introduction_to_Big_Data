import streamlit as st
import pandas as pd
import h2o
import os
from dotenv import load_dotenv
import google.generativeai as genai
import joblib


load_dotenv()


st.set_page_config(
    page_title="MindGuard | Student Stress Assistant",
    page_icon="🧠"
)


JAVA_HOME_PATH = "C:/Program Files/Java/jdk-21"
os.environ["H2O_JAVA_HOME"] = JAVA_HOME_PATH


@st.cache_resource
def get_h2o_model(model_path):
    cluster_info = h2o.cluster()

    # Check if cluster is running
    if cluster_info is None or not cluster_info.is_running():
        try:
            print(f"H2O cluster not running. Starting new one using: {os.environ.get('H2O_JAVA_HOME')}")
            h2o.init()
            print("H2O cluster successfully started.")
        except Exception as e:
            st.error("Failed to start H2O cluster. Please ensure your Java JDK is correctly set up.")
            st.error(f"Technical Error: {e}")
            return None
    else:
        print("Already connected to a running H2O cluster.")

    # Load the model
    if os.path.exists(model_path):
        print("Loading saved model...")
        model = h2o.load_model(model_path)
        print("Model successfully loaded.")
        return model
    else:
        st.error(f"Error: Model file not found at path '{model_path}'.")
        return None


def get_gemini_recommendations(stress_level_text, input_data):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # No Google API key — provide a safe local fallback recommendation instead
        return generate_local_recommendations(stress_level_text, input_data)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-2.5-pro')

        # Format key factors (based on our top-5 analysis)
        factors_text = (
            f"- Sleep Quality: {input_data['sleep_quality'][0]}/5\n"
            f"- Teacher-Student Relationship: {input_data['teacher_student_relationship'][0]}/5\n"
            f"- Future Career Concerns: {input_data['future_career_concerns'][0]}/5\n"
            f"- Depression Level: {input_data['depression'][0]}/27\n"
            f"- Academic Performance: {input_data['academic_performance'][0]}/5"
        )


        prompt = f"""
        # ROLE AND GOAL:
        You are "MindGuard", an empathetic and supportive AI assistant for students. Your primary goal is to provide safe, constructive, and encouraging stress management advice. You are NOT a doctor or therapist.

        # USER CONTEXT:
        I am a student. My current stress level was assessed as: {stress_level_text}.
        Key factors likely influencing my condition are:
        {factors_text}

        # TASK:
        Based on the provided stress level and key factors, generate a response structured into three parts:

        1.  **Support and Validation (1-2 sentences):** Acknowledge the student's feelings. Use phrases like "It is absolutely normal to feel this way..." or "It sounds like you are going through a difficult period...".

        2.  **Specific Recommendations (3 short bullet points):** Provide 3 practical and easy-to-implement pieces of advice directly related to the stress level and key factors. Advice must be action-oriented (e.g., "Try doing...", "Allocate 15 minutes for...").


        # RULES AND CONSTRAINTS:
        - **STYLE:** Friendly, calm, clear, and concise. The response MUST be in English.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Don't dump the raw API error to the user UI. Show a friendly message and fall back.
        err_text = str(e)
        # Detect common quota/rate-limit signals (simplified check)
        if '429' in err_text or 'quota' in err_text.lower() or 'rate limit' in err_text.lower():
            st.warning("The Gemini API is temporarily unavailable due to quota or rate limits. Showing local recommendations instead.")
            st.info("To enable richer AI-generated advice, add a valid `GOOGLE_API_KEY` with an appropriate quota/billing plan. See the README for details.")
        else:
            st.warning("The Gemini AI service returned an error. Showing local recommendations instead.")

        # Log the full error to the server console for debugging, but keep the UI message concise.
        print("Gemini API error:", err_text)

        # Fall back to local recommendations
        return generate_local_recommendations(stress_level_text, input_data)
    


def generate_local_recommendations(stress_level_text, input_data):
    """Return a short, safe recommendation string when Gemini API is not available.

    The content is intentionally generic and supportive (not medical advice).
    """
    # Extract a few key values safely
    try:
        sleep_q = input_data.get('sleep_quality', [None])[0]
        depression = input_data.get('depression', [None])[0]
        academic = input_data.get('academic_performance', [None])[0]
        teacher_rel = input_data.get('teacher_student_relationship', [None])[0]
    except Exception:
        sleep_q = depression = academic = teacher_rel = None

    support = (
        "It sounds like you are going through a challenging time — that's understandable. "
        "You're taking a good step by checking in on your wellbeing."
    )

    bullets = []
    # Tailor bullets simply based on a few factors
    if sleep_q is not None and sleep_q <= 2:
        bullets.append("Try a 15-minute wind-down routine before bed to improve sleep quality.")
    else:
        bullets.append("Maintain regular sleep routines — consistent bed and wake times help a lot.")

    if academic is not None and academic <= 2:
        bullets.append("Break study tasks into 25-minute focused sessions with short breaks (Pomodoro).")
    else:
        bullets.append("Keep using structured study blocks and take short breaks to avoid burnout.")

    if depression is not None and depression >= 15:
        bullets.append("If low mood or worry persists, consider contacting campus mental health services or a trusted professional.")
    elif teacher_rel is not None and teacher_rel <= 2:
        bullets.append("Try reaching out to a trusted instructor or peer to discuss academic concerns.")
    else:
        bullets.append("Reach out to friends or family and share how you feel — social support helps.")

    # Compose markdown-style text similar to Gemini output
    md = f"**Support:** {support}\n\n**Recommendations:**\n"
    for b in bullets:
        md += f"- {b}\n"

    return md



# Prefer a lightweight scikit-learn model (created by `simple_demo.py`) when present.
SKLEARN_MODEL_PATH = "rf_stress_model.joblib"
use_sklearn = False
sklearn_model = None
if os.path.exists(SKLEARN_MODEL_PATH):
    try:
        sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
        use_sklearn = True
        print(f"Loaded scikit-learn model from '{SKLEARN_MODEL_PATH}'. Streamlit will use this model (no Java/H2O required).")
    except Exception as e:
        print(f"Failed to load scikit-learn model '{SKLEARN_MODEL_PATH}': {e}. Falling back to H2O if available.")

ml_model = None
if not use_sklearn:
    MODEL_PATH = "XGBoost_1_AutoML_1_20251102_85004"
    ml_model = get_h2o_model(MODEL_PATH)

st.title("🧠 MindGuard: Assess Your Stress Level")
st.write("Answer a few questions to get an assessment of your current stress level and personalized recommendations.")

# Proceed if either a scikit-learn model or the H2O model is available
if use_sklearn or ml_model:
    if use_sklearn:
        st.info("Using local scikit-learn model (rf_stress_model.joblib) for predictions.")
    else:
        st.info("Using H2O model for predictions.")
    st.subheader("Please rate the following factors:")
    col1, col2 = st.columns(2)
    with col1:
        # User Inputs (Key Factors based on Var. Importance)
        sleep_quality = st.slider("Sleep Quality (0-Poor, 5-Excellent)", 0, 5, 2)
        academic_performance = st.slider("Academic Performance (0-Low, 5-High)", 0, 5, 2)
        social_support = st.slider("Social Support (0-None, 3-Strong)", 0, 3, 1)
        depression = st.slider("Depression Level (0-None, 27-High)", 0, 27, 10)
    with col2:
        teacher_student_relationship = st.slider("Teacher-Student Relationship (0-Poor, 5-Excellent)", 0, 5, 2)
        future_career_concerns = st.slider("Future Career Concerns (0-None, 5-Strong)", 0, 5, 3)
        self_esteem = st.slider("Self Esteem (0-Low, 30-High)", 0, 30, 15)
        headache = st.slider("Headache Frequency (0-Never, 5-Constant)", 0, 5, 2)

    if st.button("Assess and Get Advice", type="primary"):
        # 1. Compile all 20 features (8 from user, 12 as default/mean)
        input_data = {
            'sleep_quality': [sleep_quality], 'teacher_student_relationship': [teacher_student_relationship],
            'blood_pressure': [2], 'future_career_concerns': [future_career_concerns],
            'depression': [depression], 'academic_performance': [academic_performance],
            'social_support': [social_support], 'self_esteem': [self_esteem], 'safety': [2],
            'headache': [headache], 'anxiety_level': [11], 'mental_health_history': [0],
            'breathing_problem': [2], 'noise_level': [2], 'living_conditions': [2],
            'basic_needs': [2], 'study_load': [2], 'peer_pressure': [2],
            'extracurricular_activities': [2], 'bullying': [2]
        }

        input_df = pd.DataFrame(input_data)

        # 2. Make Prediction (scikit-learn fallback or H2O)
        if use_sklearn and sklearn_model is not None:
            try:
                preds = sklearn_model.predict(input_df)
                # sklearn returns integer labels (0/1/2)
                predicted_level = float(preds[0])
            except Exception as e:
                st.error(f"Failed to predict with scikit-learn model: {e}")
                predicted_level = None
        else:
            if ml_model is None:
                st.error("No ML model available (H2O model failed to load and no scikit-learn fallback found).")
                predicted_level = None
            else:
                h2o_input_frame = h2o.H2OFrame(input_df)
                prediction = ml_model.predict(h2o_input_frame)
                predicted_level = prediction['predict'].as_data_frame().iloc[0, 0]
        stress_map = {0.0: "Low", 1.0: "Medium", 2.0: "High"}
        predicted_stress_text = stress_map.get(predicted_level, "Unknown")

        st.subheader(f"Your Assessed Stress Level: {predicted_stress_text}")

        # 3. Call Gemini AI
        with st.spinner("AI Assistant is generating your personalized advice..."):
            recommendations = get_gemini_recommendations(predicted_stress_text, input_data)
            if recommendations:
                st.markdown("---")
                st.subheader("💡 Your Personalized Recommendations:")
                st.markdown(recommendations)
else:
    st.warning("The application failed to load the model. Please check the technical error output in the terminal.")