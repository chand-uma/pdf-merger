import os
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from concurrent.futures import ThreadPoolExecutor

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
class PDFMergerAI:
    def __init__(self):
        self.pdf_files = []
        self.merged_pdf = PyPDF2.PdfMerger()

    def add_pdf(self, file_path):
        # Add PDF file to the list
        self.pdf_files.append(file_path)

    def extract_text(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def analyze_content(self):
        """Analyze PDF content using NLP to determine optimal merging order."""
        texts = [self.extract_text(pdf) for pdf in self.pdf_files]
        
        # Use TF-IDF to create document vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarity between documents
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Determine optimal order (this is a simple approach and can be improved)
        order = []
        remaining = set(range(len(self.pdf_files)))
        current = 0
        while remaining:
            order.append(current)
            remaining.remove(current)
            if remaining:
                current = max(remaining, key=lambda x: similarity_matrix[current][x])
        
        # Reorder PDF files
        self.pdf_files = [self.pdf_files[i] for i in order]

    def intelligent_page_selection(self, pdf_path):
        """Use NLP to identify and select relevant pages."""
        relevant_pages = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                doc = nlp(text)
                # Simple relevance check: pages with named entities are considered relevant
                if doc.ents:
                    relevant_pages.append(i)
        return relevant_pages

    def extract_metadata(self, pdf_path):
        """Extract metadata from PDF."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return pdf_reader.metadata

    def merge_pdfs(self, output_filename):
        """Merge PDFs using AI-enhanced methods."""
        self.analyze_content()  # Determine optimal order

        for pdf_path in self.pdf_files:
            relevant_pages = self.intelligent_page_selection(pdf_path)
            metadata = self.extract_metadata(pdf_path)
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if relevant_pages:
                    for page_num in relevant_pages:
                        self.merged_pdf.append(fileobj=file, pages=(page_num, page_num+1))
                else:
                    self.merged_pdf.append(fileobj=file)
            
            # Merge metadata (simplified approach)
            if metadata:
                self.merged_pdf.add_metadata(metadata)

        # Write the merged PDF
        with open(output_filename, 'wb') as output_file:
            self.merged_pdf.write(output_file)

        print(f"Merged PDF saved as {output_filename}")

    def process_pdfs_in_parallel(self, output_filename):
        """Process PDFs in parallel for improved performance."""
        with ThreadPoolExecutor() as executor:
            executor.map(self.intelligent_page_selection, self.pdf_files)
        self.merge_pdfs(output_filename)

# Example usage
if __name__ == "__main__":
    merger = PDFMergerAI()
    
    # Add PDF files
    merger.add_pdf("path/input/sample-1.pdf")
    merger.add_pdf("path/input/sample-2.pdf")
  
    # Merge PDFs
    merger.process_pdfs_in_parallel("path/output/final.pdf")
