import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import PyPDF2
import docx2txt
import base64
from collections import Counter
import sqlite3
import re # Import re for regex used in experience detection

# --- 1. Database Setup ---
# Connect to SQLite database (will create the file if it doesn't exist)
conn = sqlite3.connect('resume_data.db')
c = conn.cursor()

# Create a table to store resume analysis results
c.execute('''
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        match_score REAL,
        skills TEXT,
        experience_years REAL,
        education TEXT,
        word_count INTEGER
    )
''')
conn.commit()
# --- End of Database Setup ---


# Load English tokenizer, tagger, parser, NER and word vectors
@st.cache_resource
def load_nlp_model():
    """Load the spaCy model once and cache it."""
    # This model needs to be downloaded via: python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# Set page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (assuming style.css is available)
def local_css(file_name):
    """Loads CSS from a local file."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Ignore if style.css is not found

# Load the custom CSS file
# local_css("style.css") # Uncomment if you have style.css


# Sidebar
with st.sidebar:
    st.image("https://storage.googleapis.com/workspace-0f70711f-8b4e-4d94-86f1-2a93ccde5887/image/413b5e0b-c425-479e-b4e1-7934822e0865.png", use_container_width=True)
    st.title("AI Resume Analyzer")
    st.markdown("""
    **Automatically score resumes** against job descriptions using NLP and machine learning.
    """)
    
    st.markdown("---")
    st.markdown("### How to Use:")
    st.markdown("""
    1. Upload multiple resumes (PDF/DOCX)
    2. Enter/Paste the job description
    3. Click 'Analyze Resumes'
    4. View results in the **Detailed Results** tab
    """)
    st.markdown("---")
    st.markdown("### Key Features:")
    st.markdown("""
    - ✨ Smart resume parsing
    - 🎯 Semantic matching
    - 📊 Candidate scoring
    - 💾 **Data Persistence**
    """)


# --- Utility Functions (unchanged) ---

def parse_resume(file):
    """Parse resume text from PDF or DOCX files."""
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            text = docx2txt.process(file)
        except Exception as e:
            st.error(f"Error processing DOCX file: {e}")
            text = ""
    return text

def extract_skills(text):
    """Extract skills using a combination of NLP and a predefined list."""
    doc = nlp(text.lower())
    skills = []
    
    technical_skills = ["python", "java", "sql", "machine learning", "data analysis", 
                         "tensorflow", "pytorch", "javascript", "react", "node.js", 
                         "c++", "c#", "html", "css", "aws", "azure", "gcp", "docker", 
                         "kubernetes", "git", "linux", "cloud", "agile", "scrum", "devops",
                         "frontend", "backend", "full-stack", "api", "database", "web development",
                         "mobile development", "ui/ux", "tableau", "power bi", "excel", "r",
                         "git", "jira", "confluence"] 
    
    for skill in technical_skills:
        if skill in text.lower():
            skills.append(skill.capitalize())
            
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "NORP", "GPE", "LANGUAGE"]:
            if len(ent.text.split()) < 4 and ent.text.lower() not in [s.lower() for s in skills]:
                if ent.text.lower() not in ["united states", "inc", "corp", "company", "group"]:
                    skills.append(ent.text)

    return sorted(list(set(skills)))[:20] 

def calculate_similarity(job_desc, resumes):
    """Calculate TF-IDF cosine similarity between job description and resumes."""
    vectorizer = TfidfVectorizer(stop_words='english')
    documents = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    job_desc_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    
    similarity_matrix = cosine_similarity(job_desc_vec, resume_vecs)
    return similarity_matrix[0]

def analyze_resume(content):
    """Perform comprehensive resume analysis."""
    doc = nlp(content)
    word_count = len(content.split())
    experience_years = 0
    
    for match in re.finditer(r'(\d+)\s*(?:year|yr)s?\s*(?:of|exp|experience)?', content.lower()):
        try:
            years = int(match.group(1))
            experience_years = max(experience_years, years)
        except ValueError:
            pass

    education = []
    education_keywords = ["university", "college", "bachelor", "master", "ph.d.", "phd", "degree", "diploma"]
    for sent in doc.sents:
        if any(term in sent.text.lower() for term in education_keywords):
            education.append(sent.text.strip())
    
    return {
        "word_count": word_count,
        "experience_years": min(experience_years, 30),
        "education": list(set(education))[:3],
        "skills": extract_skills(content)
    }


def process_resumes(uploaded_files, job_desc):
    """Process all resumes and return analysis results."""
    results = []
    parsed_resumes = []
    
    for file in uploaded_files:
        try:
            content = parse_resume(file)
            if content:
                parsed_resumes.append(content)
                analysis = analyze_resume(content)
                results.append({
                    "filename": file.name,
                    **analysis
                })
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    if parsed_resumes and job_desc:
        similarity_scores = calculate_similarity(job_desc, parsed_resumes)
        for i in range(len(results)):
            results[i]["match_score"] = round(similarity_scores[i] * 100, 1)
            
            # --- Database Insertion/Update ---
            skills_str = ', '.join(results[i]['skills'])
            education_str = ' | '.join(results[i]['education'])
            
            c.execute('''
                INSERT OR REPLACE INTO resumes (filename, match_score, skills, experience_years, education, word_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (results[i]['filename'], results[i]['match_score'], skills_str, results[i]['experience_years'], education_str, results[i]['word_count']))
            conn.commit()
            # --- End of Database Insertion/Update ---
    
    return results

def get_download_link(df, filename="resume_analysis.csv"):
    """Generates a link to download the analysis results."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

# --- End of Utility Functions ---


# Main UI - Tabs
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("Upload resumes and a job description to automatically rank candidates. View the detailed results and historical data in the other tabs.")

# Initialize session state for analysis results
if 'analysis_df' not in st.session_state:
    st.session_state['analysis_df'] = pd.DataFrame()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🚀 Analyze Candidates", "📊 Detailed Results & Insights", "📚 Past Analyses"])

# ----------------------------------------------------------------------
# --- Tab 1: Analyze Candidates ---
# ----------------------------------------------------------------------
with tab1:
    
    # File upload section
    with st.expander("📁 Upload Files & Job Description", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Resumes (PDF or DOCX)",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="uploaded_resumes",
                help="Upload multiple resumes to analyze"
            )
        with col2:
            job_desc = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste or type the job description here...",
                key="job_description_input",
                help="The more detailed the job description, the better the matching"
            )

    # Analysis section
    if st.button("✨ START ANALYSIS", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one resume")
        elif not job_desc.strip():
            st.warning("Please enter a job description")
        else:
            with st.spinner("Analyzing resumes... This may take a few moments"):
                results = process_resumes(uploaded_files, job_desc)
                
                if results:
                    # Prepare data for display
                    df = pd.DataFrame(results).sort_values("match_score", ascending=False)
                    df = df.reset_index(drop=True)
                    df.insert(0, "Rank", range(1, len(df) + 1))
                    
                    # Store DataFrame in session state
                    st.session_state['analysis_df'] = df
                    
                    # Results summary
                    st.success(f"✅ Analysis complete! Processed {len(df)} resumes. Now switch to the 'Detailed Results & Insights' tab.")
                    
                    st.markdown("---")
                    
                    # Top candidates (Quick View in Tab 1)
                    st.subheader("🏆 Top Candidates (Quick View)")
                    
                    top_n = min(3, len(df))
                    cols = st.columns(top_n)
                    
                    for i in range(top_n):
                        with cols[i]:
                            candidate = df.iloc[i]
                            st.metric(
                                label=f"Rank #{i+1} - {candidate['filename']}",
                                value=f"{candidate['match_score']}% match",
                                help=f"Skills: {', '.join(candidate['skills'][:5])}..."
                            )
# ----------------------------------------------------------------------
# --- Tab 2: Detailed Results & Insights ---
# ----------------------------------------------------------------------
with tab2:
    if st.session_state['analysis_df'].empty:
        st.info("No analysis data available. Please upload resumes and click 'START ANALYSIS' in the first tab.")
    else:
        df = st.session_state['analysis_df']
        
        # Detailed results table
        st.subheader("📊 Detailed Analysis Table")
        st.markdown("Complete, ranked list of candidates with all extracted metrics.")
        st.dataframe(
            df.style.background_gradient(
                subset=["match_score"],
                cmap="YlGnBu"
            ).format({"match_score": "{:.1f}%"}),
            use_container_width=True,
            height=600,
        )
        
        # Download button
        st.markdown(get_download_link(df), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Insights section
        st.subheader("🔍 Key Insights")
        
        # Skill frequency analysis
        all_skills = [skill for sublist in df["skills"] for skill in sublist]
        skill_counts = Counter(all_skills)
        top_skills = skill_counts.most_common(10)
        
        col1_i, col2_i = st.columns(2)
        
        with col1_i:
            st.markdown("#### Top Skills Across All Candidates")
            skills_df = pd.DataFrame(top_skills, columns=["Skill", "Count"])
            st.bar_chart(skills_df.set_index("Skill"))
        
        with col2_i:
            st.markdown("#### Score Distribution")
            score_df = df[['match_score']].set_index(df.index)
            st.bar_chart(score_df)
        
        # Resume quality metrics
        st.markdown("#### Resume Quality Metrics")
        cols_m = st.columns(4)
        metrics = {
            "📄 Avg Words": int(df["word_count"].mean()),
            "📊 Avg Score": f"{df['match_score'].mean():.1f}%",
            "👔 Avg Experience": f"{df['experience_years'].mean():.1f} yrs",
            "🎓 Avg Education Mentions": f"{df['education'].apply(len).mean():.1f}"
        }
        
        for i, (label, value) in enumerate(metrics.items()):
            cols_m[i % 4].metric(label=label, value=value)
        
        # Duplicate detection (basic)
        st.markdown("---")
        st.subheader("🔄 Duplicate Check")
        
        if len(df) > 1:
            duplicate_candidates = []
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    score_diff = abs(df.iloc[i]["match_score"] - df.iloc[j]["match_score"])
                    length_diff = abs(df.iloc[i]["word_count"] - df.iloc[j]["word_count"])
                    
                    if score_diff < 5 and length_diff < 50:
                        duplicate_candidates.append((df.iloc[i]["filename"], df.iloc[j]["filename"]))
            
            if duplicate_candidates:
                st.warning(f"⚠️ Possible duplicates detected between: {', '.join([f'{a} & {b}' for a, b in duplicate_candidates])}")
                st.info("This is a basic check. Manual review is recommended for close matches.")
            else:
                st.success("No obvious duplicates detected")
        else:
            st.info("Need at least 2 resumes for duplicate check")


# ----------------------------------------------------------------------
# --- Tab 3: View Past Analyses ---
# ----------------------------------------------------------------------
with tab3:
    st.subheader("📚 Historical Analysis Data (Database)")
    st.markdown("Results from all previous analyses are loaded and ranked by match score.")

    try:
        df_past = pd.read_sql_query("SELECT * FROM resumes ORDER BY match_score DESC", conn)
        
        if not df_past.empty:
            # Prepare data for display
            df_past['skills'] = df_past['skills'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else [])
            df_past['education'] = df_past['education'].apply(lambda x: x.split(' | ') if isinstance(x, str) and x else [])
            
            df_past.insert(0, "Rank", range(1, len(df_past) + 1))
            
            display_cols = ['Rank', 'filename', 'match_score', 'experience_years', 'skills', 'education', 'word_count']
            
            st.dataframe(
                df_past[display_cols].style.background_gradient(
                    subset=["match_score"],
                    cmap="YlGnBu"
                ).format({"match_score": "{:.1f}%"}),
                use_container_width=True,
                height=400
            )
            
            st.markdown(get_download_link(df_past, filename="all_resume_analyses.csv"), unsafe_allow_html=True)
            
            st.markdown("---")
            st.info(f"Total historical entries in database: **{len(df_past)}**")
        else:
            st.info("No past resume analyses found in the database yet. Run an analysis in the 'Analyze Candidates' tab to save data.")
    
    except Exception as e:
        st.error(f"Error loading past analyses from database: {e}")
# --- End of View Past Analyses Tab ---


# Footer (Outside tabs)
st.markdown("---")
st.markdown("""
<small>⚡ AI Resume Analyzer - Powered by Streamlit & spaCy</small>
""", unsafe_allow_html=True)

# Function to close DB connection (for completeness, though Streamlit handles resources uniquely)
@st.cache_resource
def close_db_connection():
    conn.close()