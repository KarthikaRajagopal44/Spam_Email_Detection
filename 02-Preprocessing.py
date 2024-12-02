import re  # For regular expressions
from nltk.stem import PorterStemmer  # For stemming
from nltk.corpus import stopwords  # For stopword removal
from typing import List, Tuple  # For type hinting
import string  # For string operations
from collections import Counter  # For creating unique sets of tokens
import nltk  # For downloading stopwords

# Ensure stopwords resource is downloaded
nltk.download('stopwords')

class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))  # Initialize a set of stopwords
        self.stemmer = PorterStemmer()  # Initialize the PorterStemmer for stemming

    def check_special_char(self, ch: str) -> bool:
        """
        Checks if a character is a special character or a digit.
        Returns True if it is, otherwise False.
        """
        return ch in string.punctuation or ch.isdigit()

    def remove_special_char(self, text: Tuple[str, str]) -> Tuple[str, str]:
        """
        Removes special characters and digits from the text.
        Replaces them with a space to preserve word boundaries.
        """
        sub, mes = text
        sub = ''.join([' ' if self.check_special_char(c) else c for c in sub])
        mes = ''.join([' ' if self.check_special_char(c) else c for c in mes])
        return sub, mes

    def lowercase_conversion(self, text: Tuple[str, str]) -> Tuple[str, str]:
        """
        Converts all characters in the text to lowercase.
        """
        sub, mes = text
        return sub.lower(), mes.lower()

    def tokenize(self, text: Tuple[str, str]) -> Tuple[List[str], List[str]]:
        """
        Splits the text into individual words (tokens) based on spaces.
        """
        sub, mes = text
        return sub.split(), mes.split()

    def check_stop_words(self, word: str) -> bool:
        """
        Checks if a word is a stopword.
        """
        return word in self.stop_words

    def removal_of_stop_words(self, tokens: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
        """
        Removes stopwords from the tokenized text.
        """
        sub_tokens, mes_tokens = tokens
        sub_tokens = [word for word in sub_tokens if not self.check_stop_words(word)]
        mes_tokens = [word for word in mes_tokens if not self.check_stop_words(word)]
        return sub_tokens, mes_tokens

    def stem_words(self, tokens: Tuple[List[str], List[str]]) -> List[str]:
        """
        Stems each word in the tokenized text using PorterStemmer.
        Removes duplicates by returning a unique list of stems.
        """
        sub_tokens, mes_tokens = tokens
        unique_stems = set()

        # Stem tokens from both subject and message
        for word in sub_tokens + mes_tokens:
            unique_stems.add(self.stemmer.stem(word))

        return list(unique_stems)

# Example Usage
if __name__ == "__main__":
    # Example input: (subject, message)
    text = ("HELLO!!! This is an example subject 123.", "This is an example message with special chars!! @@#$")

    preprocessor = Preprocessing()

    # Remove special characters
    text = preprocessor.remove_special_char(text)
    print("After removing special characters:", text)

    # Convert to lowercase
    text = preprocessor.lowercase_conversion(text)
    print("After converting to lowercase:", text)

    # Tokenize
    tokens = preprocessor.tokenize(text)
    print("After tokenizing:", tokens)

    # Remove stopwords
    tokens = preprocessor.removal_of_stop_words(tokens)
    print("After removing stopwords:", tokens)

    # Stem words
    stems = preprocessor.stem_words(tokens)
    print("After stemming:", stems)
