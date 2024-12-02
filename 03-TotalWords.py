
import re  # For regular expressions
from nltk.stem import PorterStemmer  # For stemming
from nltk.corpus import stopwords  # For stopword removal
import string  # For string operations
import csv  # For reading and writing CSV files

class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))  # Initialize a set of stopwords
        self.stemmer = PorterStemmer()  # Initialize the PorterStemmer for stemming

    def remove_special_char(self, text: tuple[str, str]) -> tuple[str, str]:
        """
        Removes special characters and digits from the text.
        """
        sub, mes = text
        sub = re.sub(r'[^\w\s]', ' ', sub)  # Replace non-alphanumeric characters with spaces
        mes = re.sub(r'[^\w\s]', ' ', mes)
        return sub, mes

    def lowercase_conversion(self, text: tuple[str, str]) -> tuple[str, str]:
        """
        Converts all characters in the text to lowercase.
        """
        sub, mes = text
        return sub.lower(), mes.lower()

    def tokenize(self, text: tuple[str, str]) -> tuple[list[str], list[str]]:
        """
        Splits the text into individual words (tokens).
        """
        sub, mes = text
        return sub.split(), mes.split()

    def removal_of_stop_words(self, tokens: tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
        """
        Removes stopwords from the tokenized text.
        """
        sub_tokens, mes_tokens = tokens
        sub_tokens = [word for word in sub_tokens if word not in self.stop_words]
        mes_tokens = [word for word in mes_tokens if word not in self.stop_words]
        return sub_tokens, mes_tokens

    def stem_words(self, tokens: tuple[list[str], list[str]]) -> list[str]:
        """
        Stems each word in the tokenized text.
        Removes duplicates by returning a unique list of stems.
        """
        sub_tokens, mes_tokens = tokens
        return list({self.stemmer.stem(word) for word in sub_tokens + mes_tokens})


# Main program to process the dataset
if __name__ == "__main__":
    # Initialize the Preprocessing class
    preprocessor = Preprocessing()

    # Variables to store unique words
    unique_words = set()

    # Open the CSV file for reading
    with open("Final_Dataset.csv", "r", encoding="utf-8") as infile:
        csv_reader = csv.reader(infile)
        next(csv_reader)  # Skip the header line

        # Process each row in the dataset
        for i, row in enumerate(csv_reader):
            subject = row[0]  # First column is the subject
            message = row[1]  # Second column is the message

            # Preprocess the subject and message
            text = (subject, message)
            text = preprocessor.remove_special_char(text)
            text = preprocessor.lowercase_conversion(text)
            tokens = preprocessor.tokenize(text)
            filtered_tokens = preprocessor.removal_of_stop_words(tokens)
            stemmed_tokens = preprocessor.stem_words(filtered_tokens)

            # Add stemmed tokens to the unique words set
            unique_words.update(stemmed_tokens)

            print(f"Processed row {i + 1}")  # Print progress

    # Write unique words to a file
    with open("uniquewords.txt", "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(unique_words))  # Join words with space and write to file

    print("Unique words have been saved to uniquewords.txt.")
