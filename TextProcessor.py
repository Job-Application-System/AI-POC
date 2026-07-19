from string import punctuation
from word_tokenizer import WordTokenizer
import emoji
from langdetect import detect

class TextProcessor:
    def getOccurences(targetText_split, sourceText_split):
        """
        Counts the occurrences of each keyword from the target text in both source texts.

        Args:
            targetText_split (list[str]): List of words from the target text.
            sourceText_split (list[str]): List of words from the first source text.

        Returns:
            dict: A dictionary with keywords as keys and their total occurrences in both source texts as values.
        """
        occurences = {}
        for keyword in targetText_split:
            keyword = keyword.translate(str.maketrans('', '', punctuation))
            keyword_countInSrc = sourceText_split.count(keyword)
            occurences.update({
                keyword: keyword_countInSrc
            })
        return occurences

    def normalize(text: str) -> str:
        """
        Normalizes text by removing emojis, punctuation and converting all characters to lowercase.
        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text.
        """
        lang = detect(text)
        text = emoji.demojize(text, language=lang if lang else "en")
        text = text.translate(str.maketrans('', '', punctuation)).replace("\t", "").replace("\n", "").replace("\r", "").lower()
        return text

    def tokenize(text: str) -> list[str]:
        """
        Tokenizes text of the corpus and the sources by splitting it into words.
        """
        tokenizer = WordTokenizer()
        tokenizedText: list[str] = tokenizer.tokenize(text)
        return tokenizedText