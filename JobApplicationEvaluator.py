from SimilarityModel import SimilarityModel
from TextReader import TextReader
import os

# Define Paths
ROOT = os.path.join(os.path.realpath(os.path.abspath(os.curdir)), "data") # root directory for the data
jdPath = os.path.join(ROOT, "Job Description.txt")
resumePath = os.path.join(ROOT, "Resume _ Mock King Lun Kelvin.docx")
coverLetterPath = os.path.join(ROOT, "Cover Letter.pdf")

# Read Data from files
resume = TextReader.parseText(resumePath)
coverLetter = TextReader.parseText(coverLetterPath)
jobDescription = TextReader.parseText(jdPath)

# Initialize Model
model = SimilarityModel(
    resume=resume,
    coverLetter=coverLetter,
    jobDescription=jobDescription
)

# Evaluate the Similarity Score
similarityScore = model.evaluateFinalScore()

print(f"Similarity Score: {similarityScore}")