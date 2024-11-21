FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "src.main"]
```

# docs/source/conf.py
```python
# Configuration file for the Sphinx documentation builder.

project = 'Arabic NER'
copyright = '2024, Mossab Ibrahim'
author = 'Mossab Ibrahim'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = []

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static']
```

# docs/source/index.rst
```rst
Welcome to Arabic NER's documentation!
====================================

Arabic Named Entity Recognition (NER) is a Python library for recognizing named entities
in classical Arabic texts, with a special focus on literary works like
One Thousand and One Nights.

Features
--------

* Hybrid approach combining rule-based and CRF models
* Advanced feature extraction for classical Arabic
* Comprehensive evaluation and visualization tools
* Support for multiple entity types
* Interactive visualization dashboard

Installation
-----------

You can install the package using pip:

.. code-block:: bash

   pip install arabic-ner

Quick Start
----------

Here's a simple example of how to use the library:

.. code-block:: python

   from arabic_ner import CRFModel
   
   # Initialize model
   model = CRFModel()
   
   # Train model
   model.fit(train_tokens, train_labels)
   
   # Make predictions
   predictions = model.predict(test_tokens)

Contents
--------

.. toctree::
   :maxdepth: 2
   
   installation
   usage
   api
   examples
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

# docs/source/usage.rst
```rst
Usage Guide
==========

This guide covers the main functionality of the Arabic NER library.

Data Preparation
--------------

The library expects input data in the following format:

.. code-block:: python

   # Example data format
   tokens = [
       ['قال', 'الملك', 'شهريار'],
       ['في', 'مدينة', 'بغداد']
   ]
   
   labels = [
       ['O', 'B-PERSON', 'I-PERSON'],
       ['O', 'B-LOCATION', 'I-LOCATION']
   ]

Models
------

CRF Model
^^^^^^^^^

The CRF model combines statistical learning with advanced feature extraction:

.. code-block:: python

   from arabic_ner import CRFModel
   
   model = CRFModel()
   model.fit(train_tokens, train_labels)
   predictions = model.predict(test_tokens)

Rule-based Model
^^^^^^^^^^^^^^^

The rule-based model uses linguistic patterns and gazetteers:

.. code-block:: python

   from arabic_ner import RuleBasedModel
   
   model = RuleBasedModel()
   predictions = model.predict(test_tokens)

Evaluation
---------

Use the EntityEvaluator for comprehensive evaluation:

.. code-block:: python

   from arabic_ner import EntityEvaluator
   
   evaluator = EntityEvaluator()
   metrics = evaluator.compute_metrics(test_labels, predictions)
   report = evaluator.generate_report()

Visualization
------------

Create visualizations using the ResultVisualizer:

.. code-block:: python

   from arabic_ner import ResultVisualizer
   
   visualizer = ResultVisualizer()
   visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
   visualizer.create_interactive_dashboard(metrics, error_analysis)
```

# .dockerignore
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.env
.venv
venv/
ENV/
.idea/
.vscode/
*.log
data/*
!data/.gitkeep
