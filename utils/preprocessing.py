import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class AdvancedSkillMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
        
        # Extended skill dictionary
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift'],
            'web_frontend': ['html', 'css', 'react', 'angular', 'vue', 'typescript', 'bootstrap'],
            'web_backend': ['node.js', 'django', 'flask', 'spring', 'express', 'fastapi'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'ml_ai': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch'],
            'data_science': ['pandas', 'numpy', 'r', 'tableau', 'powerbi', 'excel'],
            'devops': ['jenkins', 'git', 'ci/cd', 'ansible', 'linux', 'bash']
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_skills_with_categories(self, text):
        """Extract skills and categorize them"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            category_skills = []
            for skill in skills:
                if skill in text_lower:
                    category_skills.append(skill)
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills
    
    def calculate_comprehensive_similarity(self, job_descriptions, resume_text):
        """Calculate enhanced similarity with multiple factors"""
        # Text similarity
        processed_jobs = [self.preprocess_text(job) for job in job_descriptions]
        processed_resume = self.preprocess_text(resume_text)
        
        all_texts = processed_jobs + [processed_resume]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        job_vectors = tfidf_matrix[:-1]
        resume_vector = tfidf_matrix[-1]
        
        text_similarities = cosine_similarity(resume_vector, job_vectors)[0]
        
        # Skill-based similarity
        resume_skills = self.extract_skills_with_categories(resume_text)
        all_resume_skills = []
        for skills in resume_skills.values():
            all_resume_skills.extend(skills)
        
        skill_similarities = []
        for job_desc in job_descriptions:
            job_skills = self.extract_skills_with_categories(job_desc)
            all_job_skills = []
            for skills in job_skills.values():
                all_job_skills.extend(skills)
            
            if all_job_skills:
                match_rate = len(set(all_resume_skills) & set(all_job_skills)) / len(all_job_skills)
                skill_similarities.append(match_rate)
            else:
                skill_similarities.append(0)
        
        # Combined score (weighted average)
        final_scores = 0.7 * text_similarities + 0.3 * np.array(skill_similarities)
        
        return final_scores