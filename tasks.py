import random
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

model_save_path = './models/your_new_model.pickle'
file_name = 'books_small_10000.json'

class sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return sentiment.NEGATIVE
        elif self.score == 3:
            return sentiment.NEUTRAL
        else:
            return sentiment.POSITIVE
        
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def evenly_dual(self):
        #filters the reviews for where reviews is negative
        negative = list(filter(lambda x: x.sentiment == sentiment.NEGATIVE, self.reviews))
        len_neg = len(negative)
        positive = list(filter(lambda x: x.sentiment == sentiment.POSITIVE, self.reviews))
        len_pos = len(positive)
        prime_len = min(len_neg, len_pos)
        if prime_len == len_pos:
            neg_shrunk = negative[:prime_len]
            self.reviews = positive + neg_shrunk
        else:
            pos_shrunk = positive[:prime_len]
            self.reviews = negative + pos_shrunk
        random.shuffle(self.reviews)


    def evenly_trio(self):
        negative = list(filter(lambda x: x.sentiment == sentiment.NEGATIVE, self.reviews))
        len_neg = len(negative)
        positive = list(filter(lambda x: x.sentiment == sentiment.POSITIVE, self.reviews))
        len_pos = len(positive)
        neutral = list(filter(lambda x: x.sentiment == sentiment.NEUTRAL, self.reviews))
        len_neu = len(neutral)
        prime_len = min(len_pos, len_neg, len_neu)
        if prime_len == len_pos:
            neu_shrunk = neutral[:prime_len]
            neg_shrunk = negative[:prime_len]
            self.reviews = positive + neg_shrunk + neu_shrunk
        elif prime_len == len_neg:
            neu_shrunk = neutral[:prime_len]
            pos_shrunk = positive[:prime_len]
            self.reviews = negative + pos_shrunk + neu_shrunk
        else:
            neg_shrunk = negative[:prime_len]
            pos_shrunk = positive[:prime_len]
            self.reviews = neutral + pos_shrunk + neg_shrunk
        random.shuffle(self.reviews)

    def get_text(self):
        return [ x.text for x in self.reviews]
    
    def get_y(self):
        return [ y.sentiment for y in self.reviews]

'''
we only need the reviews text and rating
'''
reviews = []


with open(file_name) as f:
    for line in f:
        inline = json.loads(line)
        reviews.append(Review(inline['reviewText'],inline['overall']))


train, test = train_test_split( reviews, test_size=0.33, random_state=42)
train_container_2 = ReviewContainer(train)
test_container_2 = ReviewContainer(test)


train_container_2.evenly_dual()

train_x = train_container_2.get_text()
train_y = train_container_2.get_y()

test_container_2.evenly_dual()

test_x = test_container_2.get_text()
test_y = test_container_2.get_y()

vectorizer = CountVectorizer()

train_x_vector = vectorizer.fit_transform(train_x)
test_x_vector = vectorizer.transform(test_x)

log_reg = LogisticRegression()
log_reg.fit(train_x_vector,train_y)

with open(model_save_path,'wb') as f:
    pickle.dump(log_reg, f)