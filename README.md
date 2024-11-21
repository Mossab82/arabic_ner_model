This repository contains the implementation of a hybrid Named Entity Recognition (NER) system for classical Arabic texts, specifically designed for processing One Thousand and One Nights and similar classical Arabic literature.

## Features

- Hybrid approach combining rule-based techniques with Conditional Random Fields (CRF)
- Specialized feature engineering for classical Arabic
- Support for multiple entity types (characters, locations, mythical entities, objects)
- Comprehensive evaluation metrics and visualization tools

## Installation

git clone https://github.com/Mossab82/arabic_ner_model.git
cd arabic_ner_model
pip install -r requirements.txt


## Usage

from src.models.crf_model import CRFModel
from src.config import CRF_CONFIG

# Initialize model
model = CRFModel(CRF_CONFIG)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
metrics = model.evaluate(X_test, y_test)

## Citation

If you use this code in your research, please cite:

@article{ibrahim2024named,
  title={Named Entity Recognition in Classical Arabic Literature: A Hybrid Approach Applied to One Thousand and One Nights},
  author={Ibrahim, M. and Gervás, P. and Méndez, G.},
  journal={Journal Not Specified},
  year={2024}
}

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# License
MIT License

Copyright (c) 2024 Mossab Ibrahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
