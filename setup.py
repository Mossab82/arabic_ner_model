from setuptools import setup, find_packages

setup(
    name="arabic-ner-model",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
    ],
    author="Mossab Ibrahim",
    author_email="mibrahim@ucm.es",
    description="Classical Arabic Named Entity Recognition",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mossab82/arabic_ner_model",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
