from setuptools import setup, find_packages
import os

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name='soundq2',
    version='0.1.0',
    description='Code of Soundq2',
    author='Aiden Chang',
    author_email='your.email@example.com',
    url='https://github.com/aiden200/SoundQ2',
    packages=find_packages(),
    install_requires=read_requirements(),
    # extras_require={
    #     'dev': [
    #         'pytest>=7.0',
    #         'black>=24.2.0',
    #         # Add other development dependencies here
    #     ],
    # },
    python_requires='>=3.10',
)