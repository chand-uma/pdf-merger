# PDF Merger AI

An intelligent PDF merging tool that uses Natural Language Processing (NLP) and Machine Learning to optimize PDF combinations.

## Features

- Smart document ordering using TF-IDF and cosine similarity
- Intelligent page selection based on content relevance
- Metadata preservation during merging
- Parallel processing for improved performance
- Named entity recognition for content analysis

## Requirements

- Python 3.7+
- Required packages:
  - PyPDF2
  - nltk
  - scikit-learn
  - spacy
  - transformers
  - requests

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-merger
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

```python
from pdfmerger import PDFMergerAI

# Initialize the merger
merger = PDFMergerAI()

# Add PDF files
merger.add_pdf("path/to/first.pdf")
merger.add_pdf("path/to/second.pdf")

# Merge PDFs with intelligent processing
merger.process_pdfs_in_parallel("output.pdf")
```

## How It Works

1. **Content Analysis**: Uses NLP to analyze PDF content and determine optimal merging order
2. **Smart Selection**: Identifies relevant pages based on content importance
3. **Metadata Preservation**: Maintains document metadata during merging
4. **Parallel Processing**: Utilizes multi-threading for improved performance

## License

This project is licensed under the MIT License.
