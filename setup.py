from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="emotion_classification",
    version="1.0.0",
    author="Tonmoy & Ioanna",
    packages=find_packages(),
    install_requires=parse_requirements('./config/requirements.txt'),
    python_requires='==3.12.3',
)
