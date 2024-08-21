from setuptools import setup, find_packages
import os

base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, "README.md")) as f:
    long_description = f.read()

setup(
  name="veda_lib",
  version="0.0.5",
  author="Vishal Maurya",
  author_email="vishallmaurya210@gmail.com",
  description="veda_lib is a Python library designed to streamline the data preprocessing and cleaning workflow for machine learning projects. It offers a comprehensive set of tools to handle common data preparation tasks",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/vishallmaurya/VEDA",
  license="Apache License 2.0",
  packages=find_packages(where='src'),
  package_dir={'': 'src'},
  classifiers=[
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
  ],
  keywords=["Automated Data Preprocessing", "Data Cleaning", "Data Balancing", "Machine Learning", 
            "Data Transformation", "Feature Engineering", "Data Wrangling", "Data Preparation",
            "Exploratory Data Analysis"], 
  project_urls={
      "Bug Tracker": "https://github.com/vishallmaurya/VEDA/issues",
  },
  python_requires=">=3.9",
  install_requires=[
      "numpy>=1.21.0",
      "pandas>=1.3.0",
      "scikit-learn>=0.24.0",
      "imbalanced-learn>=0.8.0",
      "tensorflow>=2.4.0",
      "umap-learn>=0.5.0",
      "optuna>=2.7.0",
      "statsmodels>=0.12.0",
      "diptest>=0.1.0",
  ],
  include_package_data=True,  # Includes files from MANIFEST.in
  zip_safe=False,  # Whether the package can be distributed as a .zip file
) 
    
    