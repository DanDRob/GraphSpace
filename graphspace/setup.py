from setuptools import setup, find_packages

setup(
    name="graphspace",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "torch>=1.8.0",
        "torch-geometric>=2.0.0",
        "networkx>=2.5",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",

        # NLP and machine learning
        "transformers>=4.5.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.0",
        "openai>=1.0.0",

        # Web framework
        "flask>=2.0.0",
        "flask-cors>=3.0.10",

        # Document processing
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "pdfminer.six>=20220524",
        "langchain-text-splitters>=0.0.1",
        "markdown>=3.0.1",
        "bs4>=0.0.1",
        "filetype>=1.2.0",

        # Google integration
        "google-api-python-client>=2.100.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=1.0.0",

        # Calendar integration
        "icalendar>=4.0.0",
        "recurring_ical_events>=1.0.0",

        # Utilities
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "matplotlib>=3.3.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.7",
    author="",
    author_email="",
    description="A knowledge graph-based productivity assistant",
    keywords="knowledge graph, productivity, RAG",
    url="",
)
