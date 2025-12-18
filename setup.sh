#!/bin/bash

# Create necessary directories
mkdir -p .streamlit
mkdir -p data
mkdir -p utils
mkdir -p assets

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

echo "Setup completed successfully!"