import pickle

with open('./models/log_reg_countVectorizer.pickle','rb') as f:
    loaded_log_CountVectorizer = pickle.load(f)

with open('./models/vectorized_text.pickle','rb') as f:
    vectorizer = pickle.load(f)


string_input = input("Enter the reviews?_ ")
string = [string_input]

model_input = vectorizer.transform(string)

output = loaded_log_CountVectorizer.predict(model_input)

print(output)