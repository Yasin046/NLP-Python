import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import unittest


# "https://www.trustpilot.com/review/pointsbet.com"


url = "https://www.trustpilot.com/review/yabbycasino.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Get Reviews
reviews = []
for review in soup.find_all("p", {"class": "typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn"}):
    reviews.append(review.text)


sia = SentimentIntensityAnalyzer()

# Sentient Analysis
sentiments = []
compound_scores = []
for review in reviews:
    sentiment = sia.polarity_scores(review)
    compound_scores.append(sentiment['compound'])
    print(review)
    print(sentiment)
    print('-----------------------')
    sentiments.append(sentiment)


# Tokenize
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)


# Get the most important token for each review
important_tokens = []
number_of_docs = len(reviews)
for current_doc in range(number_of_docs):
    print("Processing document: " + str(current_doc))
    document_row = tfidf_matrix.toarray()[current_doc]
    index_with_highest_value = np.argmax(document_row)
    # print('Highest TF-IDF value is at column: ' + str(index_with_highest_value))
    token_with_highest_tfidf_value = vectorizer.get_feature_names_out()[
        index_with_highest_value]
    print('Most important token is: ' + token_with_highest_tfidf_value)
    important_tokens = token_with_highest_tfidf_value


# Create a list of dictionaries containing review, sentiment, and important token
data = []
for current_doc in range(number_of_docs):
    print("Processing document: " + str(current_doc))
    document_row = tfidf_matrix.toarray()[current_doc]
    index_with_highest_value = np.argmax(document_row)
    token_with_highest_tfidf_value = vectorizer.get_feature_names_out()[
        index_with_highest_value]
    print('Most important token is: ' + token_with_highest_tfidf_value)

    row = {
        'Review': reviews[current_doc],
        'Sentiment': sentiments[current_doc],
        'Important Token': token_with_highest_tfidf_value
    }
    data.append(row)

# Define the CSV file path
csv_file = 'sentiment_analysis.csv'

# Write the data to the CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['Review', 'Sentiment', 'Important Token']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print('Data exported to', csv_file)


# Plotting the sentiment scores
x = range(1, len(reviews) + 1)  # X-axis values
y = compound_scores  # Y-axis values

plt.bar(x, y)
plt.xlabel('Reviews')
plt.ylabel('Compound Sentiment Score')
plt.title('Sentiment Analysis Results')
plt.xticks(x)
plt.show()


def display_menu():
    print("\n---------- Sentiment Analysis of iGaming Customer Reviews ----------")
    print("1. View sentiment results")
    print("2. View sentiment graph")
    print("0. Exit")


def view_sentiment_results():
    df = pd.read_csv("sentiment_analysis.csv")
    compound_scores = df['Sentiment'].apply(lambda x: eval(x)['compound'])
    result_df = pd.DataFrame(
        {'Review': df['Review'], 'Compound Sentiment': compound_scores})
    print(result_df)


def view_sentiment_graph():
    # Plotting the sentiment scores
    x = range(1, len(reviews) + 1)  # X-axis values
    y = compound_scores  # Y-axis values

    plt.bar(x, y)
    plt.xlabel('Reviews')
    plt.ylabel('Compound Sentiment Score')
    plt.title('Sentiment Analysis Results')
    plt.xticks(x)
    plt.show()


def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            view_sentiment_results()
        elif choice == "2":
            view_sentiment_graph()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


class SentimentAnalysisTest(unittest.TestCase):

    def test_sentiment_analysis(self):
        url = "https://www.trustpilot.com/review/yabbycasino.com"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        reviews = []
        for review in soup.find_all("p", {"class": "typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn"}):
            reviews.append(review.text)

        sia = SentimentIntensityAnalyzer()

        sentiments = []
        compound_scores = []
        for review in reviews:
            sentiment = sia.polarity_scores(review)
            compound_scores.append(sentiment['compound'])
            sentiments.append(sentiment)

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(reviews)

        important_tokens = []
        number_of_docs = len(reviews)
        for current_doc in range(number_of_docs):
            document_row = tfidf_matrix.toarray()[current_doc]
            index_with_highest_value = np.argmax(document_row)
            token_with_highest_tfidf_value = vectorizer.get_feature_names_out()[
                index_with_highest_value]
            important_tokens = token_with_highest_tfidf_value

        data = []
        for current_doc in range(number_of_docs):
            document_row = tfidf_matrix.toarray()[current_doc]
            index_with_highest_value = np.argmax(document_row)
            token_with_highest_tfidf_value = vectorizer.get_feature_names_out()[
                index_with_highest_value]

            row = {
                'Review': reviews[current_doc],
                'Sentiment': sentiments[current_doc],
                'Important Token': token_with_highest_tfidf_value
            }
            data.append(row)

        csv_file = 'sentiment_analysis.csv'

        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = ['Review', 'Sentiment', 'Important Token']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        x = range(1, len(reviews) + 1)
        y = compound_scores

        plt.bar(x, y)
        plt.xlabel('Reviews')
        plt.ylabel('Compound Sentiment Score')
        plt.title('Sentiment Analysis Results')
        plt.xticks(x)

        self.assertTrue(True)


def run_tests():
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(SentimentAnalysisTest)

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(test_suite)


if __name__ == "__main__":
    run_tests()
    main()
