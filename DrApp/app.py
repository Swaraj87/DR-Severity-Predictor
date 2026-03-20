import streamlit as st
from PIL import Image

from src.config import Config, GRADE_INFO
from src.inference import Inference
from src.gradcam import GradCAMEngine

# 1. Page CONFIGURATION

#This is used to define tab title, icons, and to set layouts to use full width of the page
#This must be in start of stramlit command
st.set_page_config(
    page_title="Diabetic Retinopahty Screener",
    page_icon="👁️",
    layout='wide'
)

# 2.Backend Initialization
# ***: @st.cache_resource ensures the heavy Pytorch/XGB model are loaded
# only once and stays in the ram. Without this the app would freeze to reload
#all 7 models every time user click a button.

@st.cache_resource
def load_backend():
    with st.spinner("Loading Pytorch and XGBoost models..."):
        #Main inference pipeline and load weights
        inference_engine = Inference(model_dir="savemodels")
        # Passing the Loaded CNN directly to Grad-Cam engine
        gradcam_engine = GradCAMEngine(managers=inference_engine.manager,
                                       device=inference_engine.device)
    return inference_engine, gradcam_engine

#loading models into memory
infer_engine, gradcam_engine = load_backend()

# 3 Main UI Header

st.title("Diabetic Retinopathy Severity Detection")
st.markdown("""
Welcome to the DR screening. We are using a proven stacking ensemble of CNNs
combined with **XGBoost** to classify retinal fundus images and provide an
explainalbe AI (Grad-CAM) Visuliaztion.
""")
st.divider()

# 4. SIDEBAR AND DATA INPUT
# Isolating the user inputs to sidebar for main dashboard to reamina clean
st.sidebar.header("Paitent Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Retinal Fundus Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

# After Image is uploaded.
if uploaded_file is not None:
    # Uploaded into PIL RGB image for pipeline
    pil_image = Image.open(uploaded_file).convert('RGB')

    # Splitting the main UI into 2 columns for Image and Results
    col1, col2 = st.columns([1, 1.5])

    # For the Image and Control
    with col1:
        st.subheader("Uploaded Scan")
        st.image(pil_image, use_column_width=True, caption="Raw Fundus Image")

        # A button to start the model and prevent form automatic action, giveing user control
        analyze_button = st.button("Run Ensemble Analysis", type="primary", use_container_width=True)

    # For Inference Results
    if analyze_button:
        with col2:
            st.subheader("Diagonstic Resutls")

            with st.spinner("Running throught CNNs and XGBoost..."):
                # Travesing Full Workflow
                results = infer_engine.predict(pil_image)

                # Extract Features: Prioritzing the SMOTE Ensemble
                if "ensemble" in results:
                    final_grade = results['ensemble']['grade']
                    confidence = results["ensemble"]["confidence"]
                    model_used = "SOMTE ENSEMBLE"
                else:
                    final_grade = results["consensus_grade"]
                    confidence = 1.0
                    model_used = "Majority Vote Consensus"
                
                # Human readable info from config.py based on prediction
                grade_details = GRADE_INFO[final_grade]

                # Rendering Dashboard-style metric cards
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Predicted Severity", value=grade_details['name'])
                m2.metric(label="Confidence", value=f"{confidence* 100:.2f}%")
                m3.metric(label="Model Used:", value=model_used)

                #Advise display based on severity
                st.info(f"**Urgency:** {grade_details['urgency']} \n**Recommended Action:** {grade_details['action']}")
        st.divider()

    # 5. EXPLAINABLE AI (GRAD-CAM)
        st.subheader("🔍 Explainable AI: Grad-CAM Feature Activation")
        st.markdown("*Heatmaps indicate the specific vascular regions the CNNs focused on to make this prediction.*")
        
        with st.spinner("Generating feature heatmaps..."):
            
            # Generate heatmaps targeting the specific predicted severity grade
            overlays = gradcam_engine.generate(pil_image, target_grade=final_grade)
            
            if overlays:
                # Dynamically generate Streamlit columns based on how many models succeeded
                cam_cols = st.columns(len(overlays))
                
                # Iterate through the dictionary of returned heatmaps and display them side-by-side
                for idx, (model_name, overlay_img) in enumerate(overlays.items()):
                    with cam_cols[idx]:
                        if overlay_img is not None:
                            st.image(overlay_img, caption=f"{model_name.upper()} Activation", use_container_width=True)
                        else:
                            st.warning(f"Grad-CAM failed for {model_name}")
            else:
                st.error("Grad-CAM library not available or failed to generate.")

# Default state when no image is uploaded
else:
    st.info("👈 Please upload a retinal fundus image from the sidebar to begin.")