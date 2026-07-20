from gensim.models import Word2Vec
import numpy as np
from TextProcessor import TextProcessor
from difflib import SequenceMatcher
from collections import Counter

class SimilarityModel:
    jobDescription: str
    resume: str
    coverLetter: str

    tokenizedCorpus: list[str]
    tokenizedSource: list[str]

    SCORE_THRESHOLD = 0.1  # Threshold for individual similarity scores

    def __init__(self, jobDescription: str, resume: str, coverLetter: str):
        self.jobDescription = TextProcessor.normalize(jobDescription)
        self.resume = TextProcessor.normalize(resume)
        self.coverLetter = TextProcessor.normalize(coverLetter)
        # Tokenization
        self.tokenizedCorpus = TextProcessor.tokenize(self.jobDescription)
        self.tokenizedSource = TextProcessor.tokenize(self.resume) + TextProcessor.tokenize(self.coverLetter)
    
    def simiilarityFromOccurences(self) -> float:
        """
        Calculates the similarity between the target text and the source texts based on the occurrences of keywords.

        Returns:
            float: The average similarity score normalized by the length of the tokenized source.
        """
        similarity_scores = []
        occurences = TextProcessor.getOccurences(self.tokenizedCorpus, self.tokenizedSource)
        for key, value in occurences.items():
            similarity_scores.append(value)
        if not self.tokenizedSource:
            return 0.0
        return sum(similarity_scores) / len(self.tokenizedSource)

    def simiilarityFromNumOfWords(self, occurences_source, occurences_target, targetText_split):        
        """
        Calculates the similarity between the target text and the source texts based on the number of words.

        Args:
            occurences_source (dict): A dictionary with keywords as keys and their occurrences in the source texts as values.
            occurences_target (dict): A dictionary with keywords as keys and their occurrences in the target text as values.
            targetText_split (list): A list of words in the target text.

        Returns:
            float: The similarity score normalized by the total number of words in the target text.
        """
        totalNumOfWords_jobDesc = len(targetText_split) #number of words in target text
        if totalNumOfWords_jobDesc == 0:
            return 0.0
        valueSum = 0
        for occurence_source in occurences_source:
            key = occurence_source
            value = occurences_source[occurence_source]
            if (key in occurences_target):
                if (value == occurences_target[key]):
                    valueSum += value
            else:
                continue
            if (value > 1):
                diff = (value - 1)
                totalNumOfWords_jobDesc += diff #total number of words should be enlarged accordingly
                valueSum += diff
            elif (value == 1):
                valueSum += 1
            else:
                continue
        return valueSum / totalNumOfWords_jobDesc #similarity (ranging from 0 to 1)
    
    def similarityRatio(self) -> float:
        """
        Calculates the similarity between the target text (job description) and the source texts (resume + cover letter) based on the ratio of the number of common words.

        Returns:
            float: The similarity score normalized by the length of the tokenized source.
        """
        return SequenceMatcher(None, self.jobDescription, self.resume + self.coverLetter).ratio()
    
    def similarityByCharacter(self):
        """
        Calculates the similarity between the target (job description) and the source texts (resume + cover letter) based on character occurrences.

        Returns:
            float: The similarity score normalized by the total number of characters in both the source and target texts.
        """
        occurences_fromSource = {}
        occurences_toCompare = {}
        
        # extract characters and occurences
        # resume
        for c in self.resume:
            if (c in occurences_fromSource):
                occurences_fromSource.update({
                    c: occurences_fromSource[c]+ 1
                })
            else:
                occurences_fromSource.update({
                    c: 1
                })
        # cover letter
        for c in self.coverLetter:
            if (c in occurences_fromSource):
                occurences_fromSource.update({
                    c: occurences_fromSource[c]+ 1
                })
            else:
                occurences_fromSource.update({
                    c: 1
                })
        # job description
        for c in self.jobDescription:
            if (c in occurences_toCompare):
                occurences_toCompare.update({
                    c: occurences_toCompare[c]+ 1
                })
            else:
                occurences_toCompare.update({
                    c: 1
                })
        
        #finding number of matching characters
        similarity = 0
        matchingChars = 0
        totalCharacters_ToCompare = len(occurences_toCompare)
        totalCharacters_FromSource = len(occurences_fromSource)
        if totalCharacters_ToCompare + totalCharacters_FromSource == 0:
            return 0.0
        #matching characters to Job Description
        for occurence in occurences_toCompare:
            if (occurence in occurences_fromSource):
                matchingChars += occurences_fromSource[occurence]
        #calculate similarity
        similarity = (2 * matchingChars) / (totalCharacters_ToCompare + totalCharacters_FromSource)
        if (abs(similarity) > 1):
            similarity = pow(similarity, -1) #invert fraction
        return abs(similarity)
    
    def similarityByCommonality(self):
        """
        Calculates the similarity between the target text and the source texts based on the commonality of words.

        Returns:
            float: The similarity score normalized by the number of common words.
        """
        common = Counter(self.tokenizedCorpus) & Counter(self.tokenizedSource)
        return self.simiilarityFromNumOfWords(common, common, self.tokenizedCorpus)
    
    def similarityByCosineSimilarity(self):
        def get_vector(text, model):
            """
            Generates a vector representation of the given text using the provided Word2Vec model.

            Args:
                text (list[str]): The text to be converted into a vector.
                model (Word2Vec): The Word2Vec model used to generate the vector.

            Returns:
                np.ndarray: The vector representation of the text.
            """
            vector = [model.wv[word] for word in text if word in model.wv]
            if not vector:
                return np.zeros(model.vector_size)
            return sum(vector) / len(vector)
        
        model = Word2Vec(
            sentences=[self.tokenizedSource + self.tokenizedCorpus],
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
        vectorizedSource = np.array([get_vector(self.tokenizedSource, model)])
        vectorizedCorpus = np.array([get_vector(self.tokenizedCorpus, model)])
        denominator = np.linalg.norm(vectorizedSource) * np.linalg.norm(vectorizedCorpus)
        if denominator == 0:
            return 0.0
        similarity = np.dot(vectorizedSource, vectorizedCorpus.T) / denominator
        return np.squeeze(similarity)
    
    def evaluateFinalScore(self):
        """
        Evaluates the final similarity score by combining the scores from all the similarity metrics.

        Returns:
            float: The final similarity score.
        """
        similarities = list(self.componentScores().values())
        filtered_similarities = [sim for sim in similarities if sim >= self.SCORE_THRESHOLD]
        if not filtered_similarities:
            return 0.0
        return np.mean(filtered_similarities) # reject individual invalid ones below threshold

    def componentScores(self) -> dict[str, float]:
        """
        Returns each similarity component used by evaluateFinalScore.
        """
        occurences_source = TextProcessor.getOccurences(self.tokenizedSource, self.tokenizedSource)
        occurences_target = TextProcessor.getOccurences(self.tokenizedCorpus, self.tokenizedCorpus)
        return {
            "similarity_from_occurrences": float(self.simiilarityFromOccurences()),
            "similarity_from_num_of_words": float(
                self.simiilarityFromNumOfWords(occurences_source, occurences_target, self.tokenizedCorpus)
            ),
            "similarity_ratio": float(self.similarityRatio()),
            "similarity_by_character": float(self.similarityByCharacter()),
            "similarity_by_commonality": float(self.similarityByCommonality()),
            "similarity_by_cosine": float(self.similarityByCosineSimilarity()),
        }
