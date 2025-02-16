import streamlit as st
import re
import json
from PyPDF2 import PdfReader
from docx import Document
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import os
from groq import Groq

# Initialize Groq API client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Initialize NLP components
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    # st.error("Please install the SpaCy English model: 'python -m spacy download en_core_web_sm'")
    # st.stop()

# Initialize models
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize T5 question generator with proper tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("t5-base")
question_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt"
)

def extract_text(file):
    """Extract text from various file formats"""
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    elif file.name.endswith('.txt'):
        return file.read().decode()
    return ""

def extract_contact_info(text):
    """Extract phone numbers and emails using regex"""
    phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    phones = re.findall(phone_pattern, text)
    emails = re.findall(email_pattern, text)
    
    return {
        'phone': ', '.join(phones) if phones else 'Not found',
        'email': ', '.join(emails) if emails else 'Not found'
    }


def extract_name(text):
    """Extract candidate name using SpaCy NER"""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return "Not found"

def analyze_sections(text):
    """Parse resume sections using rule-based approach"""
    sections = {
        'experience': [],
        'skills': [],
        'education': [],
        'certifications': []
    }
    
    current_section = None
    section_keywords = {
        'experience': ['experience', 'work history', 'employment'],
        'skills': ['skills', 'competencies', 'technologies'],
        'education': ['education', 'academic background'],
        'certifications': ['certifications', 'licenses', 'courses']
    }
    
    for line in text.split('\n'):
        line_lower = line.strip().lower()
        
        # Detect section headers
        for section, keywords in section_keywords.items():
            if any(keyword in line_lower for keyword in keywords):
                current_section = section
                break
        else:
            if current_section and line.strip():
                sections[current_section].append(line.strip())
    
    return {k: '\n'.join(v) if v else 'Not found' for k, v in sections.items()}

def calculate_similarity(resume_text, jd_text):
    """Calculate semantic similarity between resume and JD"""
    embeddings = similarity_model.encode([resume_text, jd_text])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item() * 100

def generate_interview_questions(resume_text, jd_text):
    """Generate interview questions using Groq API"""
    input_text = f"Generate interview questions based on the resume and job description.Here is the resume: {resume_text}\n and here is the Job Description:{jd_text} Give me concise to the point questions only. Not description of resume or Job Description."
    
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": input_text}
        ],
        model="deepseek-r1-distill-llama-70b"
    )
    return response.choices[0].message.content

# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Main Application
st.title("AI-Powered Resume Analyzer 🧠")
st.markdown("""
    Upload a candidate's resume and paste the job description to get:
    - Candidate profile analysis
    - Job requirement matching
    - Automated interview questions
""")

# File Upload and JD Input
with st.container():
    col1, col2 = st.columns([2, 3])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF/DOCX/TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, Word, Text"
        )
    
    with col2:
        jd_input = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the complete job description here..."
        )

if st.button("Process Resume"):
    if uploaded_file and jd_input:
        resume_text = extract_text(uploaded_file)
        
        if resume_text:
            # Candidate Profile Section
            st.header("👤 Candidate Profile")
            profile_col1, profile_col2 = st.columns([1, 2])
            
            with profile_col1:
                st.subheader("Basic Information")
                name = extract_name(resume_text)
                contact = extract_contact_info(resume_text)
                
                st.markdown(f"""
                    **Name:** {name}  
                    **Phone:** {contact['phone']}  
                    **Email:** {contact['email']}
                """)
            
            with profile_col2:
                st.subheader("Professional Summary")
                sections = analyze_sections(resume_text)
                
                exp_col, edu_col = st.columns(2)
                with exp_col:
                    with st.expander("Work Experience"):
                        st.write(sections['experience'])
                
                with edu_col:
                    with st.expander("Education"):
                        st.write(sections['education'])
                
                skills_col, cert_col = st.columns(2)
                with skills_col:
                    with st.expander("Skills"):
                        st.write(sections['skills'])
                
                with cert_col:
                    with st.expander("Certifications"):
                        st.write(sections['certifications'])
    
            # Job Matching Analysis
            st.header("📊 Job Compatibility Analysis")
            match_score = calculate_similarity(resume_text, jd_input)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Match Percentage", f"{match_score:.1f}%")
            
            with col2:
                st.progress(match_score/100)
                st.caption("Semantic similarity score between resume content and job description")
    
            # Interview Questions
            st.header("❓ Suggested Interview Questions")
            questions = generate_interview_questions(resume_text, jd_input)
            
            if questions:
                st.write(questions)
                # cleaned_questions = questions.replace("\\n", "\n").split("\n")
                # for i, q in enumerate(cleaned_questions[:5]):
                #     st.markdown(f"{i+1}. {q.strip()}")
            else:
                st.warning("Could not generate questions. Please try with more detailed inputs.")
    
    else:
        st.info("👆 Please upload a resume and enter a job description to begin analysis")

# Footer
st.markdown("---")
st.markdown("Built with ♥ using [Streamlit](https://streamlit.io) | [Hugging Face](https://huggingface.co) | [Spacy](https://spacy.io) | FAISS | Groq AI")
