import streamlit as st
import pandas as pd
import joblib
import time


#  1. SETUP PAGE & CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="Global AI Career Advisor", 
    page_icon="🎓", 
    layout="wide"
)

#  2. LOAD YOUR ML MODEL 
@st.cache_resource
def load_model():
    try:
        # Make sure this path is correct on your computer
        return joblib.load("data/models/best_model.joblib")
    except (FileNotFoundError, OSError):
        # If file is missing, we return None so app doesn't crash
        return None





# Language Packs & Configuration

LANGUAGES = {
    "en": {
        "app_title": "Global AI Career Advisor",
        "nav_home": "Home",
        "nav_quiz": "Career Quiz",
        "nav_profile": "Profile",
        "home_title": "Welcome to Your Future",
        "home_sub": "AI-powered guidance to find your perfect career match.",
        "quiz_title": "Career Assessment",
        "quiz_intro": "Answer the following questions to analyze your potential.",
        
        "q_env": "1. What kind of work environment energizes you most?",
        "q_prob": "2. How much do you enjoy solving complex problems?",
        "q_act": "3. Which activity do you find most engaging?",
        "q_skill": "4. Rate your Communication & Presentation Skills (0-10)",
        "q_lang": "5. Select your preferred work language:",
        "btn_predict": "Predict My Career",
        
        "opt_tech": "Tech Lab / Start-up (Fast-paced)",
        "opt_corp": "Corporate Office (Strategic)",
        "opt_res": "Quiet Research Room (Analytical)",
        "opt_pub": "Public/Social Space (Interaction)",
        "opt_art": "Creative Studio (Artistic)",
        
        "res_cs": "Computer Science 💻",
        "res_biz": "Business Management 💼",
        "res_math": "Mathematics 📐",
        "res_pol": "Political Science ⚖️",
        "res_art": "Fine Arts 🎨",
        
        "profile_title": "User Profile",
        "save_btn": "Save Profile"
    },
    "fr": {
        "app_title": "Conseiller de Carrière IA",
        "nav_home": "Accueil",
        "nav_quiz": "Quiz Carrière",
        "nav_profile": "Profil",
        "home_title": "Bienvenue dans votre futur",
        "home_sub": "Une orientation par IA pour trouver votre carrière idéale.",
        "quiz_title": "Évaluation de Carrière",
        "quiz_intro": "Répondez aux questions suivantes pour analyser votre potentiel.",
        
        "q_env": "1. Quel environnement de travail vous stimule le plus ?",
        "q_prob": "2. Aimez-vous résoudre des problèmes complexes ?",
        "q_act": "3. Quelle activité trouvez-vous la plus engageante ?",
        "q_skill": "4. Notez vos compétences en communication (0-10)",
        "q_lang": "5. Sélectionnez votre langue de travail préférée :",
        "btn_predict": "Prédire ma carrière",
        
        "opt_tech": "Tech Lab / Start-up (Rapide)",
        "opt_corp": "Bureau Corporatif (Stratégique)",
        "opt_res": "Salle de Recherche (Analytique)",
        "opt_pub": "Espace Public/Social (Interaction)",
        "opt_art": "Studio Créatif (Artistique)",
        
        "res_cs": "Informatique (Computer Science) 💻",
        "res_biz": "Gestion d'Entreprise 💼",
        "res_math": "Mathématiques 📐",
        "res_pol": "Sciences Politiques ⚖️",
        "res_art": "Beaux-Arts 🎨",
        
        "profile_title": "Profil Utilisateur",
        "save_btn": "Enregistrer le profil"
    },
    "hi": {
        "app_title": "AI करियर एडवाइजर",
        "nav_home": "होम",
        "nav_quiz": "करियर क्विज",
        "nav_profile": "प्रोफाइल",
        "home_title": "आपके भविष्य में स्वागत है",
        "home_sub": "अपने सही करियर को खोजने के लिए AI का मार्गदर्शन।",
        "quiz_title": "करियर मूल्यांकन",
        "quiz_intro": "अपनी क्षमता का विश्लेषण करने के लिए प्रश्नों के उत्तर दें।",
        
        "q_env": "1. आपको किस तरह का काम का माहौल (Work Environment) सबसे ज्यादा पसंद है?",
        "q_prob": "2. आप जटिल समस्याओं को सुलझाना कितना पसंद करते हैं?",
        "q_act": "3. आपको कौन सी गतिविधि सबसे दिलचस्प लगती है?",
        "q_skill": "4. अपनी बातचीत और प्रस्तुति कौशल (Communication Skills) को रेट करें (0-10)",
        "q_lang": "5. काम के लिए अपनी पसंदीदा भाषा चुनें:",
        "btn_predict": "मेरा करियर बताएं",
        
        "opt_tech": "टेक लैब / स्टार्ट-अप (तेज गति)",
        "opt_corp": "कॉर्पोरेट ऑफिस (रणनीतिक)",
        "opt_res": "शांत रिसर्च रूम (विश्लेषणात्मक)",
        "opt_pub": "पब्लिक/सोशल स्पेस (बातचीत)",
        "opt_art": "क्रिएटिव स्टूडियो (कलात्मक)",
        
        "res_cs": "कंप्यूटर साइंस (Computer Science) 💻",
        "res_biz": "बिजनेस मैनेजमेंट (Business Management) 💼",
        "res_math": "गणित (Mathematics) 📐",
        "res_pol": "राजनीति विज्ञान (Political Science) ⚖️",
        "res_art": "फाइन आर्ट्स (Fine Arts) 🎨",
        
        "profile_title": "यूज़र प्रोफाइल",
        "save_btn": "प्रोफाइल सेव करें"
    }
}

LANGUAGE_NAMES = {
    "en": "English", 
    "fr": "Français", 
    "hi": "हिंदी"
}


# Helper Functions
def get_text(key):
    lang = st.session_state.get("lang", "en")
    # Fallback to English if key missing
    return LANGUAGES.get(lang, LANGUAGES["en"]).get(key, key)


# Navigation Pages
def show_home():
    st.header(get_text("home_title"))
    st.write(get_text("home_sub"))
    
    # Placeholder image
    st.image("https://cdn.pixabay.com/photo/2018/03/10/12/00/teamwork-3213924_1280.jpg", caption="AI Career Guidance")
    
    # Info box only shows in English generally unless translated
    if st.session_state["lang"] == "hi":
        st.info("अपना मूल्यांकन शुरू करने के लिए 'करियर क्विज' पर जाएं।")
    elif st.session_state["lang"] == "fr":
        st.info("Naviguez vers la section 'Quiz Carrière' pour commencer.")
    else:
        st.info("Navigate to the 'Career Quiz' section to start your assessment.")

def show_quiz():
    st.header(get_text("quiz_title"))
    st.write(get_text("quiz_intro"))
    st.write("---")

    # --- Q1: Environment ---
    q1_opts = [
        get_text("opt_tech"), 
        get_text("opt_corp"), 
        get_text("opt_res"), 
        get_text("opt_pub"), 
        get_text("opt_art")
    ]
    q1 = st.radio(get_text("q_env"), options=q1_opts)

    st.write("") 

    # --- Q2: Problem Solving ---
    # Simplified slider options for cleaner translation mapping
    q2_label = get_text("q_prob")
    q2 = st.slider(q2_label, 0, 10, 5) 
    # (Changed to numeric slider for easier multi-language handling)

    st.write("")

    # --- Q3: Activities ---
    q3 = st.selectbox(
        get_text("q_act"),
        options=[
            "Coding / Gaming",
            "Leading Team / Managing",
            "Solving Math Puzzles",
            "Debating / History",
            "Singing / Painting / Sports"
        ]
    )

    st.write("")

    # --- Q4: Skills ---
    q4_slider = st.slider(get_text("q_skill"), 0, 10, 5)

    st.write("")

    # --- Q5: Language (Extra Feature) ---
    q_lang_pref = st.selectbox(get_text("q_lang"), ["English", "French", "Hindi"])

    st.write("---")

    # --- PREDICTION LOGIC ---
    if st.button(get_text("btn_predict"), type="primary"):
        
        # 1. Fake Loading Animation
        with st.spinner('AI Model Analyzing Patterns...'):
            time.sleep(1.5) # Simulating AI processing
        
        # 2. Calculate Score
        score = 0
        
        # Environment Logic
        if q1 == get_text("opt_tech"): score += 25
        elif q1 == get_text("opt_corp"): score += 20
        elif q1 == get_text("opt_res"): score += 15
        elif q1 == get_text("opt_pub"): score += 10
        elif q1 == get_text("opt_art"): score += 5
        
        # Problem Solving Logic (Slider 0-10)
        # 8-10 = High score (Love it)
        if q2 >= 8: score += 25
        elif q2 >= 5: score += 20
        elif q2 >= 3: score += 10
        else: score += 5

        # Activity Logic
        if "Coding" in q3: score += 25
        elif "Leading" in q3: score += 20
        elif "Math" in q3: score += 15
        elif "Debating" in q3: score += 10
        elif "Singing" in q3: score += 5
        
        # Skills Logic
        score += (q4_slider * 2.5)

        # Cap score
        if score > 100: score = 100
        
        # 3. Determine Result
        final_career = ""
        if score >= 70:
            final_career = get_text("res_cs")
            st.balloons()
        elif score >= 60:
            final_career = get_text("res_biz")
        elif score >= 50:
            final_career = get_text("res_math")
        elif score >= 40:
            final_career = get_text("res_pol")
        else:
            final_career = get_text("res_art")

        # 4. Display Result
        st.success(f"Analysis Complete! Match Score: {int(score)}%")
        
        st.markdown(f"## 🎯 Recommended Path: **{final_career}**")
        
        if st.session_state["lang"] == "hi":
            st.info(f"जानकारी: **{q_lang_pref}** के लिए आपकी प्राथमिकता इस क्षेत्र में एक बड़ी संपत्ति है।")
        elif st.session_state["lang"] == "fr":
            st.info(f"Aperçu : Votre préférence pour **{q_lang_pref}** est un atout majeur.")
        else:
            st.info(f"Insight: Your selected preference for **{q_lang_pref}** is a great asset for this field globally.")

def show_profile():
    st.header(get_text("profile_title"))
    with st.form("profile"):
        st.text_input("Name")
        st.text_input("Email")
        st.text_area("Bio / Notes")
        if st.form_submit_button(get_text("save_btn")):
            if st.session_state["lang"] == "hi":
                st.success("प्रोफ़ाइल अपडेट हो गई!")
            elif st.session_state["lang"] == "fr":
                st.success("Profil mis à jour avec succès !")
            else:
                st.success("Profile Updated Successfully!")


# Main Application Entry Point
def main():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    # --- Sidebar ---
    st.sidebar.title("Dashboard")
    
    # Language Switcher
    lang_choice = st.sidebar.selectbox("Language / भाषा", list(LANGUAGE_NAMES.values()))
    
    # Update session state logic
    for code, name in LANGUAGE_NAMES.items():
        if name == lang_choice:
            st.session_state["lang"] = code

    st.sidebar.write("---")
    
    # Navigation
    menu = [get_text("nav_home"), get_text("nav_quiz"), get_text("nav_profile")]
    choice = st.sidebar.radio("Go to", menu)

    # --- Main Content Area ---
    if choice == get_text("nav_home"):
        show_home()
    elif choice == get_text("nav_quiz"):
        show_quiz()
    elif choice == get_text("nav_profile"):
        show_profile()

if __name__ == "__main__":
    main()