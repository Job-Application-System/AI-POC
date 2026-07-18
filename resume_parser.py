from TextReader import TextReader
from TextProcessor import TextProcessor
from pathlib import Path

def fetch_resume(filepath):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    match Path(filepath).suffix:
        case ".pdf":
            return TextReader.readPDF(filepath)
        case ".docx" | ".doc":
            return TextReader.readWord(filepath)
        case ".txt":
            return TextReader.readTxt(filepath)
        case _:
            raise NotImplementedError(f"Unsupported file format: {Path(filepath).suffix}")

def preprocess_resume(resume):
    return TextProcessor.normalize(resume)

def main():
    # Example usage
    SAMPLE_DIR = "AI-POC/sample_resumes"
    for filename in Path(SAMPLE_DIR).iterdir():
        if filename.is_file():
            try:
                resume_text = fetch_resume(str(filename))
                normalized_resume = preprocess_resume(resume_text)
                print(f"Processed {filename.name}:")
                print(normalized_resume)
            except Exception as e:
                print(f"Error processing {filename.name}: {e}")