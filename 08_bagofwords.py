# -*- coding: utf-8 -*-
"""08 - BagOfWords.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16K9eNawK7Oli4ZnUm0r1nLcTiWRuTYW_
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile 08-BagOfWords.py
# import csv
# 
# class BagOfWords:
#     def transform(self, processed_data):
#         """
#         This function creates a Bag of Words (BoW) representation of the data.
# 
#         Steps:
#         1. Read unique words from a file.
#         2. Process the input data (processed_data) and count the occurrences of each unique word.
#         3. Save the BoW representation to a CSV file.
#         """
# 
#         # Step 1: Reading the unique words from "unique_words.txt"
#         unique_words = []  # List to store unique words
#         with open("05 - unique words.txt", "r") as in_file:
#             for line in in_file:
#                 unique_words.append(line.strip())  # Add each word to the unique_words list
# 
#         print(f"Unique words: {len(unique_words)}")  # Print the count of unique words
# 
#         # Step 2: Writing the columns (unique words) in the output "BagOfWords.csv"
#         with open("08 - BagOfWords.csv", mode="w", newline='') as out_file:
#             writer = csv.writer(out_file)
# 
#             # Write the header (unique words)
#             writer.writerow(unique_words)
# 
#             # Step 3: Creating the Bag of Words file
#             for data in processed_data:
#                 word_count = {}  # Dictionary to store word counts for the current sentence
# 
#                 # Count the occurrences of words in the current sentence
#                 for word in data:
#                     word_count[word] = word_count.get(word, 0) + 1
# 
#                 # Write the word counts for each unique word in the CSV file
#                 row = []
#                 for word in unique_words:
#                     if word in word_count:
#                         row.append(word_count[word])
#                     else:
#                         row.append(0)
# 
#                 writer.writerow(row)  # Write the row to the CSV file
# 
#                 print(f"Processed sentence {processed_data.index(data) + 1}")
# 
#

!python /content/08-BagOfWords.py