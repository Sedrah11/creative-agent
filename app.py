import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Creative Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache Hugging Face model so it loads only once
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2", cache_dir="./models")

generator = load_model()

# Sidebar for instructions / info
with st.sidebar:
    st.header("📝 About Creative Agent")
    st.write(
        """
        Enter your product or campaign details below, and this tool will generate
        3 unique marketing campaign ideas for you!
        
        **Tips for best results:**
        - Be clear with product description.
        - Include target audience and keywords.
        - Keep each field concise.
        """
    )
    st.markdown("---")
    st.write("Made with ❤️ by **Sedrah AboSamrah**")

# Main page header with custom styling
st.markdown(
    """
    <div style="background-color:#4B8BBE; padding:15px; border-radius:10px">
        <h1 style="color:white; text-align:center;">🧠 Creative Agent</h1>
        <p style="color:white; text-align:center;">Your AI-powered Creative Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Input section
st.markdown("### Enter Your Campaign Details")
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Campaign Details",
            height=250,
            placeholder="""Campaign Name: 
Product Description: 
Target Audience: 
Keywords:"""
        )
    
    with col2:
        st.markdown("#### Quick Example")
        st.write(
            """
            ```
            Campaign Name: SuperClean Detergent
            Product Description: Eco-friendly detergent for all fabrics
            Target Audience: Young adults who care about sustainability
            Keywords: eco, clean, gentle
            ```
            """
        )

st.markdown("<br>", unsafe_allow_html=True)

# Generate ideas button
if st.button("✨ Generate Ideas"):
    if user_input.strip():
        with st.spinner("Generating creative ideas..."):
            try:
                prompt = (
                    "Generate 3 unique marketing campaign ideas with a title and 1-sentence description each. "
                    "Use the following product details:\n" + user_input
                )
                
                outputs = generator(
                    prompt,
                    max_length=150,
                    num_return_sequences=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.9
                )

                st.markdown("### ✍️ Generated Ideas")
                for i, out in enumerate(outputs):
                    st.markdown(
                        """
                        <div style="background-color:#F1F1F1; padding:10px; border-radius:8px; margin-bottom:10px">
                        <b>Idea {idx}:</b> {text}
                        </div>
                        """.format(idx=i+1, text=out['generated_text'].strip()),
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter campaign details before generating.")
