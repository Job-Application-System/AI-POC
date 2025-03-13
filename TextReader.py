from PyPDF2 import PdfReader
import docx2txt
import os
from spire.doc import *
from spire.doc.common import *

class TextReader:
    def readPDF(filepath):
        """
        Reads a PDF file and extracts text from each page.

        Args:
            filepath (str): The path to the PDF file.

        Returns:
            list: A list of strings, each containing the text from a page of the PDF.
        
        Author:
            Kelvin Mock
        """
        reader = PdfReader(filepath)
        pages = reader.pages
        number_of_pages = len(pages)
        texts = []
        for page in pages:
            text = page.extract_text()
            texts.append(text)
        return texts
    
    def readWord(filepath):
        """
        Reads a Word document and extracts text.

        Args:
            filepath (str): The path to the Word document.

        Returns:
            str: The extracted text from the Word document.
        
        Author:
            Kelvin Mock
        """
        # extract text
        if (filepath.endswith('.docx')):
            text = docx2txt.process(filepath)
        elif (filepath.endswith('.doc')):
            document = Document()
            document.LoadFromFile(filepath)
            text = document.GetText().lstrip('evaluation warning: the document was created with spire.doc for python.')
        return text
    
    def readTxt(filepath):
        """
        Reads a text file and extracts text line by line.

        Args:
            filepath (str): The path to the text file.

        Returns:
            list: A list of strings, each containing a line of text from the file.
        
        Author:
            Kelvin Mock
        """
        file = open(filepath, "r")
        texts = []
        for line in file:
            texts.append(line)
        return texts
    
    def parseText(filepath):
        """
        Parses the text from a file based on its extension.

        Args:
            filepath (str): The path to the file.

        Returns:
            str: The extracted and lowercased text from the file, or None if the file format is unsupported.
        """
        if (filepath.endswith(".pdf")):
            texts = TextReader.readPDF(filepath)
            texts_str = ''
            for page in texts:
                texts_str += page.lower()
            return texts_str
        elif (filepath.endswith('.doc') or filepath.endswith('docx')):
            return TextReader.readWord(filepath).lower()
        elif (filepath.endswith('.txt')):
            texts = TextReader.readTxt(filepath)
            texts_str = ''
            for line in texts:
                texts_str += line.lower()
            return texts_str
        else:
            return None #invalid filepath or file format