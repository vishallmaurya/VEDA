from setuptools import setup, find_packages
import os

base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, "README.md")) as f:
    long_description = f.read()

setup(
  name="VEDA",
  version="0.1.0",
  author="Vishal Maurya",
  author_email="vishallmaurya210@gmail.com",
  description="A Python library for automated preprocessing and cleaning of data",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/vishallmaurya/VEDA",
  license="Apache License 2.0",
  packages=find_packages(),
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
  ],
  keywords=["automated efficient data cleaning", "automated efficient preprocessing",
            "python library", "veda"],
  project_urls={
      "Bug Tracker": "https://github.com/vishallmaurya/VEDA/issues",
  },
  python_requires=">=3.6",
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
    
    