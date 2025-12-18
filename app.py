import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import ssl

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# Download NLTK data
download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="AI Skill Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .match-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .job-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 10px 0;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .missing-skill {
        background-color: #f8d7da;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

class SkillMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text using keyword matching"""
        common_skills = {
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
            'vue', 'node', 'express', 'django', 'flask', 'fastapi', 'mongodb',
            'mysql', 'postgresql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'jenkins', 'machine learning', 'deep learning', 'nlp',
            'computer vision', 'data analysis', 'pandas', 'numpy', 'tensorflow',
            'pytorch', 'scikit-learn', 'tableau', 'powerbi', 'excel', 'typescript',
            'rest api', 'graphql', 'microservices', 'linux', 'bash', 'shell scripting',
            'agile', 'scrum', 'ci/cd', 'terraform', 'ansible', 'prometheus', 'grafana',
            'spring boot', 'hibernate', 'android', 'kotlin', 'swift', 'ios',
            'php', 'laravel', 'wordpress', 'shopify', 'mern stack', 'mean stack'
        }
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_similarity(self, job_descriptions, resume_text):
        """Calculate similarity between job descriptions and resume"""
        # Preprocess all texts
        processed_jobs = [self.preprocess_text(job) for job in job_descriptions]
        processed_resume = self.preprocess_text(resume_text)
        
        # Combine for vectorization
        all_texts = processed_jobs + [processed_resume]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        job_vectors = tfidf_matrix[:-1]
        resume_vector = tfidf_matrix[-1]
        
        similarities = cosine_similarity(resume_vector, job_vectors)
        
        return similarities[0]
    
    def get_match_analysis(self, job_description, resume_text):
        """Get detailed match analysis"""
        job_skills = self.extract_skills(job_description)
        resume_skills = self.extract_skills(resume_text)
        
        matching_skills = set(job_skills) & set(resume_skills)
        missing_skills = set(job_skills) - set(resume_skills)
        
        skill_match_rate = len(matching_skills) / len(job_skills) if job_skills else 0
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'skill_match_rate': skill_match_rate,
            'total_job_skills': len(job_skills),
            'matching_skills_count': len(matching_skills)
        }

@st.cache_data
def load_sample_data():
    """Load sample job and resume data"""
    # Sample job data
    jobs_data = {
        'Job Title': [
            'Data Scientist',
            'Frontend Developer',
            'Backend Developer',
            'ML Engineer',
            'Full Stack Developer',
            'DevOps Engineer',
            'Android Developer',
            'Java Developer',
            'Python Developer',
            'React Native Developer'
        ],
        'Company': [
            'Tech Solutions Inc', 'Digital Innovations', 'Web Services Co', 
            'AI Technologies', 'Software Creations', 'Cloud Systems',
            'Mobile Apps Ltd', 'Enterprise Solutions', 'Data Systems', 'App Developers'
        ],
        'Description': [
            'Looking for Data Scientist with strong Python skills, machine learning experience, pandas, numpy, SQL. Knowledge of TensorFlow or PyTorch required.',
            'Seeking Frontend Developer proficient in JavaScript, React, HTML, CSS. Experience with responsive design and modern frameworks.',
            'Backend Developer needed with expertise in Java, Spring Boot, Microservices, SQL. Experience with high-traffic systems.',
            'Machine Learning Engineer required with deep learning, NLP, computer vision experience. Proficiency in Python, PyTorch, TensorFlow.',
            'Full Stack Developer with React, Node.js, MongoDB, and AWS experience. Must have experience in startup environment.',
            'DevOps Engineer with Docker, Kubernetes, AWS, CI/CD, Jenkins experience. Strong scripting skills.',
            'Android Developer with Kotlin/Java experience. Knowledge of Android SDK, Material Design.',
            'Java Developer with Spring Framework, Hibernate, REST APIs. Experience with enterprise applications.',
            'Python Developer with Django/Flask experience. Strong in algorithms and data structures.',
            'React Native Developer for cross-platform mobile apps. Experience with Redux, TypeScript.'
        ],
        'Required Skills': [
            'python,machine learning,pandas,numpy,sql,tensorflow',
            'javascript,react,html,css,responsive design',
            'java,spring boot,microservices,sql,hibernate',
            'python,machine learning,deep learning,nlp,computer vision,pytorch',
            'react,node.js,mongodb,aws,full stack',
            'docker,kubernetes,aws,ci/cd,jenkins,scripting',
            'android,kotlin,java,mobile development',
            'java,spring framework,hibernate,rest apis',
            'python,django,flask,algorithms,data structures',
            'react native,javascript,redux,typescript'
        ],
        'Location': [
            'Bangalore', 'Hyderabad', 'Pune', 'Delhi', 'Chennai',
            'Mumbai', 'Gurgaon', 'Noida', 'Kolkata', 'Ahmedabad'
        ],
        'Salary': [
            '₹12-18 LPA', '₹8-12 LPA', '₹10-15 LPA', '₹15-22 LPA', '₹9-14 LPA',
            '₹11-16 LPA', '₹10-14 LPA', '₹8-13 LPA', '₹7-11 LPA', '₹9-13 LPA'
        ],
        'Experience': [
            '3-6 years', '2-4 years', '3-5 years', '4-7 years', '2-5 years',
            '3-6 years', '2-4 years', '2-5 years', '1-4 years', '2-4 years'
        ]
    }
    
    # Sample resumes
    resumes_data = {
        'Resume Text': [
            """Experienced Data Scientist with 4 years experience. Strong in Python, machine learning, pandas, numpy, SQL. 
            Worked on e-commerce data analysis. Skilled in TensorFlow and statistical modeling. Education: B.Tech in Computer Science.""",
            
            """Frontend Developer with 3 years experience. Specializing in JavaScript, React, HTML5, CSS3. 
            Built responsive web applications for startups. Experience with Redux and modern frontend tools.""",
            
            """Backend Developer with Java expertise. 4 years experience. Strong in Spring Boot, Microservices, Hibernate. 
            Worked on banking applications. Knowledge of SQL and system design.""",
            
            """Machine Learning Engineer focused on computer vision. 3 years experience. 
            Proficient in PyTorch, TensorFlow, and Python. Experience with text recognition projects.""",
            
            """Full Stack Developer with MERN stack experience. Built e-commerce platforms. 
            Strong in React, Node.js, MongoDB, and AWS deployment."""
        ],
        'Title': [
            'Senior Data Scientist',
            'Frontend Developer',
            'Java Backend Developer', 
            'ML Engineer',
            'Full Stack Developer'
        ]
    }
    
    jobs_df = pd.DataFrame(jobs_data)
    resumes_df = pd.DataFrame(resumes_data)
    
    return jobs_df, resumes_df

def show_skill_matching():
    st.header("🎯 Skill Matching")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Resume")
        resume_input = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="""Enter your resume text here. Include your skills, experience, education, and projects.

Example:
Software Developer with 3 years experience.
Skills: Java, Spring Boot, Microservices, SQL, AWS.
Education: B.Tech in Computer Science.
Projects: Built payment gateway integration."""
        )
        
        # Sample resume selector
        st.subheader("🎲 Use Sample Resume")
        sample_resumes = st.session_state.resumes_df['Resume Text'].tolist()
        sample_titles = st.session_state.resumes_df['Title'].tolist()
        
        sample_options = [f"{title}" for title in sample_titles]
        selected_sample = st.selectbox("Choose a sample resume:", ["Select..."] + sample_options)
        
        if selected_sample != "Select...":
            sample_index = sample_options.index(selected_sample)
            resume_input = sample_resumes[sample_index]
            st.text_area("Sample Resume Preview:", value=resume_input, height=150, key="sample_preview")
    
    with col2:
        st.subheader("📊 Matching Results")
        
        if resume_input:
            with st.spinner('Analyzing your skills and finding matches...'):
                # Calculate similarities
                similarities = st.session_state.matcher.calculate_similarity(
                    st.session_state.jobs_df['Description'].tolist(),
                    resume_input
                )
                
                # Add similarity scores to jobs dataframe
                results_df = st.session_state.jobs_df.copy()
                results_df['Match Score'] = (similarities * 100).round(2)
                results_df = results_df.sort_values('Match Score', ascending=False)
                
                # Display progress
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                
            # Display results
            st.success(f"Found {len(results_df)} potential job matches!")
            
            for idx, row in results_df.iterrows():
                # Determine match color
                match_color = "#2ecc71" if row['Match Score'] > 70 else "#f39c12" if row['Match Score'] > 50 else "#e74c3c"
                
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{row['Job Title']} - {row['Company']}</h3>
                        <p style="color: {match_color}; font-size: 1.2rem; font-weight: bold;">
                            Match Score: {row['Match Score']}%
                        </p>
                        <p><strong>📍 Location:</strong> {row['Location']} | <strong>💰 Salary:</strong> {row['Salary']}</p>
                        <p><strong>📅 Experience:</strong> {row['Experience']}</p>
                        <p><strong>Description:</strong> {row['Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed analysis
                    with st.expander("View Detailed Skill Analysis"):
                        analysis = st.session_state.matcher.get_match_analysis(
                            row['Description'], resume_input
                        )
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.success(f"✅ Matching Skills ({analysis['matching_skills_count']} found)")
                            if analysis['matching_skills']:
                                for skill in analysis['matching_skills']:
                                    st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No matching skills found")
                        
                        with col_b:
                            st.error(f"❌ Missing Skills ({len(analysis['missing_skills'])} needed)")
                            if analysis['missing_skills']:
                                for skill in analysis['missing_skills']:
                                    st.markdown(f'<div class="missing-skill">{skill}</div>', unsafe_allow_html=True)
                            else:
                                st.success("All required skills matched!")
                        
                        st.info(f"**Skill Match Rate:** {analysis['skill_match_rate']*100:.1f}%")
            
            # Visualization
            st.subheader("📈 Match Score Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['#2ecc71' if x > 70 else '#f39c12' if x > 50 else '#e74c3c' for x in results_df['Match Score']]
            bars = ax.barh(results_df['Job Title'] + ' - ' + results_df['Company'], results_df['Match Score'], color=colors)
            ax.set_xlabel('Match Score (%)')
            ax.set_title('Job Match Scores')
            ax.bar_label(bars, fmt='%.1f%%')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("👆 Please enter your resume text or select a sample resume to see matching jobs.")

def show_job_search():
    st.header("🔍 Job Search")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("Search jobs by title, skills, or company:")
    
    with col2:
        location_filter = st.selectbox("Filter by city:", ["All Locations"] + st.session_state.jobs_df['Location'].unique().tolist())
    
    with col3:
        experience_filter = st.selectbox("Experience level:", ["All Levels"] + st.session_state.jobs_df['Experience'].unique().tolist())
    
    filtered_jobs = st.session_state.jobs_df.copy()
    
    if search_term:
        filtered_jobs = filtered_jobs[
            filtered_jobs['Job Title'].str.contains(search_term, case=False) |
            filtered_jobs['Description'].str.contains(search_term, case=False) |
            filtered_jobs['Company'].str.contains(search_term, case=False) |
            filtered_jobs['Required Skills'].str.contains(search_term, case=False)
        ]
    
    if location_filter != "All Locations":
        filtered_jobs = filtered_jobs[filtered_jobs['Location'] == location_filter]
    
    if experience_filter != "All Levels":
        filtered_jobs = filtered_jobs[filtered_jobs['Experience'] == experience_filter]
    
    if len(filtered_jobs) == 0:
        st.warning("No jobs found matching your criteria. Try broadening your search.")
    else:
        st.success(f"Found {len(filtered_jobs)} jobs matching your criteria.")
        
        for idx, row in filtered_jobs.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <h3>{row['Job Title']} - {row['Company']}</h3>
                    <p><strong>📍 Location:</strong> {row['Location']} | <strong>💰 Salary:</strong> {row['Salary']}</p>
                    <p><strong>📅 Experience:</strong> {row['Experience']}</p>
                    <p><strong>Description:</strong> {row['Description']}</p>
                    <p><strong>Required Skills:</strong> {row['Required Skills']}</p>
                </div>
                """, unsafe_allow_html=True)

def show_resume_analysis():
    st.header("📊 Resume Analysis")
    
    resume_text = st.text_area("Paste your resume for analysis:", height=300,
                              placeholder="Paste your resume text here to analyze your skills and get improvement suggestions...")
    
    if resume_text:
        matcher = st.session_state.matcher
        
        with st.spinner('Analyzing your resume...'):
            # Extract skills
            skills = matcher.extract_skills(resume_text)
            
            # Skill categories
            skill_categories = {
                'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'kotlin', 'swift'],
                'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask'],
                'Mobile Development': ['android', 'react native', 'flutter', 'ios', 'mobile development'],
                'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
                'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ci/cd'],
                'Data Science & ML': ['python', 'machine learning', 'deep learning', 'nlp', 'computer vision', 
                                    'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
                'Java Ecosystem': ['java', 'spring boot', 'hibernate', 'microservices', 'rest apis'],
                'Tools & Other': ['git', 'linux', 'bash', 'agile', 'scrum', 'tableau', 'powerbi', 'excel']
            }
            
            categorized_skills = {}
            for category, category_skills in skill_categories.items():
                found_skills = [skill for skill in skills if skill in category_skills]
                if found_skills:
                    categorized_skills[category] = found_skills
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠️ Detected Skills")
            st.metric("Total Skills Detected", len(skills))
            
            for category, category_skills in categorized_skills.items():
                with st.expander(f"{category} ({len(category_skills)} skills)"):
                    for skill in category_skills:
                        st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Skill Distribution")
            if skills:
                # Create skill count by category
                category_counts = {category: len(skills_list) for category, skills_list in categorized_skills.items()}
                
                fig, ax = plt.subplots(figsize=(8, 6))
                if category_counts:
                    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a']
                    ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', 
                          startangle=90, colors=colors[:len(category_counts)])
                    ax.axis('equal')
                    ax.set_title('Skill Distribution')
                    st.pyplot(fig)
            else:
                st.info("No skills detected. Make sure to include technical skills in your resume.")
        
        # Resume suggestions
        st.subheader("💡 Improvement Suggestions")
        
        suggestion_count = 0
        
        if len(skills) < 5:
            st.warning("🔸 Consider adding more technical skills to your resume")
            suggestion_count += 1
        
        if 'experience' not in resume_text.lower() and 'work' not in resume_text.lower():
            st.warning("🔸 Add more details about your work experience and projects")
            suggestion_count += 1
        
        if 'education' not in resume_text.lower() and 'degree' not in resume_text.lower():
            st.warning("🔸 Include your educational background")
            suggestion_count += 1
        
        if 'project' not in resume_text.lower():
            st.warning("🔸 Add details about your projects and achievements")
            suggestion_count += 1
        
        # Check for high-demand skills
        high_demand_skills = ['java', 'python', 'react', 'aws', 'sql', 'javascript']
        missing_high_demand = [skill for skill in high_demand_skills if skill not in skills]
        if missing_high_demand:
            st.info(f"🔸 High-demand skills: Consider adding {', '.join(missing_high_demand)}")
        
        if suggestion_count == 0:
            st.success("🎉 Your resume looks good! It includes key sections and technical skills.")

def show_about():
    st.header("ℹ️ About AI Skill Matcher")
    
    st.markdown("""
    ### 🚀 How It Works
    
    This AI-powered skill matcher uses advanced Natural Language Processing (NLP) techniques to:
    
    1. **Text Processing**: Cleans and preprocesses resume and job description text
    2. **Skill Extraction**: Identifies key technical skills using keyword matching
    3. **Similarity Analysis**: Uses TF-IDF vectorization and cosine similarity
    4. **Match Scoring**: Calculates compatibility scores between resumes and jobs
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **ML Libraries**: Scikit-learn, NLTK
    - **NLP**: TF-IDF, Cosine Similarity
    - **Data Processing**: Pandas, NumPy
    
    ### 📈 Features
    
    - Real-time skill matching
    - Detailed match analysis
    - Skill gap identification
    - Interactive visualizations
    - Sample data for testing
    
    ### 🎯 Use Cases
    
    - Job seekers finding compatible roles
    - Recruiters identifying suitable candidates
    - Career guidance and skill development
    - Resume optimization and improvement
    
    ### 🔧 How to Use
    
    1. Go to **Skill Matching** tab
    2. Enter your resume text or use a sample resume
    3. View matching jobs with scores
    4. Analyze skill gaps and matches
    5. Use insights to improve your resume
    """)

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🔍 AI-Powered Skill Matcher</h1>', unsafe_allow_html=True)
    st.markdown("### Bridge the Gap Between Job Seekers and Roles")
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = SkillMatcher()
    if 'jobs_df' not in st.session_state:
        st.session_state.jobs_df, st.session_state.resumes_df = load_sample_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Skill Matching", "Job Search", "Resume Analysis", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quick Start:**
    1. Go to **Skill Matching**
    2. Enter your resume
    3. View job matches
    4. Analyze skill gaps
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.success("""
    **Popular Skills:**
    - Java & Spring Boot
    - Python & Django/Flask
    - React & Node.js
    - AWS & DevOps
    - Data Science & ML
    """)
    
    if app_mode == "Skill Matching":
        show_skill_matching()
    elif app_mode == "Job Search":
        show_job_search()
    elif app_mode == "Resume Analysis":
        show_resume_analysis()
    else:
        show_about()

if __name__ == "__main__":
    main()