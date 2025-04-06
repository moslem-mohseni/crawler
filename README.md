# Clone the repository and navigate to project folder
git init
git add .
git commit -m "Initial project structure"

# Create and activate virtual environment
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download fa_core_news_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Build and start Docker containers
docker-compose up -d

# Initialize database
python scripts/init_database.py


# ایجاد جداول
python scripts/init_database.py

# ایجاد جداول و بارگذاری داده‌های اولیه
python scripts/init_database.py --seed

# بازسازی کامل پایگاه داده
python scripts/init_database.py --recreate --seed