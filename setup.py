from setuptools import find_packages, setup

setup(
    name="mcqgenerator",
    version="0.0.1",
    author="Suraj",
    author_email="suryanshp1@gmail.com",
    install_requires=["langchain-groq", "langchain_core", "langchain", "python-dotenv", "streamlit", "PyPDF2"],
    packages=find_packages(),
)