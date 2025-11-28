import streamlit as st
from medical_qa import *

# Page configuration
st.set_page_config(
    page_title="🏥 AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .response-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏥 AI Medical Assistant (RAG Pipeline)</h1>', unsafe_allow_html=True)

# Context and disclaimer
st.markdown("""
<div class="sub-header">
    <strong>Context:</strong> This system retrieves information specifically from the <code>mtsamples</code> dataset. 
    It is designed to be a safe, citation-aware assistant.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Disclaimer:</strong> This is an academic project for NLP. Do not use for real medical advice.
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
st.sidebar.header("📝 Example Queries")

example_queries = [
    "What are the surgical options for carpal tunnel syndrome?",
    "Describe the post-operative care for a tonsillectomy.",
    "What medications are mentioned for treating hypertension?",
    "What technique was used for the laparoscopic cholecystectomy?",
    "Describe the incision made for the carpal tunnel release.",
    "What anesthesia was used for the colonoscopy procedure?",
    "What were the findings of the MRI of the lumbar spine?",
    "How was the patient positioned for the right total knee arthroplasty?",
    "What sutures were used to close the fascia in the hernia repair?",
    "Describe the findings during the cystoscopy."
]

# Let user select an example or type their own
use_example = st.sidebar.checkbox("Use example query")
if use_example:
    selected_query = st.sidebar.selectbox("Select an example:", example_queries)
    query = selected_query
else:
    query = st.sidebar.text_area("Or type your medical query:", height=100)

# Main query input
if not use_example:
    query = st.text_input(
        "🔍 Medical Query:",
        placeholder="e.g., What are the common symptoms of allergic rhinitis?",
        value=query
    )

# Generate response button
if st.button("🚀 Analyze", type="primary"):
    if query.strip():
        with st.spinner("🔄 Processing your medical query..."):
            try:
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                
                # Display response
                st.markdown("### 💡 Medical Response")
                st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
                
                # Show context information
                if response.get('context'):
                    st.markdown("### 📚 Source Context")
                    for i, doc in enumerate(response['context'][:3], 1):
                        row_id = doc.metadata.get('row', 'Unknown')
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        with st.expander(f"📄 Source Document {i} (Row {row_id})"):
                            st.text(doc.page_content)
                            
            except Exception as e:
                st.error(f"⚠️ Error: Something went wrong in the pipeline. Details: {str(e)}")
    else:
        st.warning("⚠️ Please enter a medical query before analyzing.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🏥 AI Medical Assistant | RAG Pipeline | Powered by Gemini API</p>
    <p>Built with Streamlit • Dataset: mtsamples</p>
</div>
""", unsafe_allow_html=True)
