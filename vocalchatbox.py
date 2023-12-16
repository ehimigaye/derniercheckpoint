import nltk
import streamlit as st
import speech_recognition as sr

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

with open(r'C:\Users\toshiba\Desktop\datascience\chatbox\mobydick.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

sentences = sent_tokenize(data)


def preprocess(sentence):
    words = word_tokenize(sentence)

    words = [word.lower() for word in words if
             word.lower() not in stopwords.words('english') and word not in string.punctuation]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

corpus = [preprocess(sentence) for sentence in sentences]

def get_most_relevant_sentence(query):
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio_text = r.listen(source)
        st.info("Transcribing...")
        try:
            text = r.recognize_google(audio_text)
            return text
        except:
            return "Sorry, I did not get that."

def chatbot(question):

    if question == "ecrite" :
        most_relevant_sentence = get_most_relevant_sentence(question)
        return most_relevant_sentence
    else :
        text = transcribe_speech()


def main():
    st.title("Chatbot")
    st.write("Bonjour, merci de me donner un texte  ou un vocal.")

    question = st.radio("Ecrire ou Vocale: ", ('ecrite', 'vocal'))
    if (question == 'ecrite'):
        question = st.text_input("You:")
        if st.button("Ecrivez le texte :"):
            response = chatbot(question)
            st.write("Chatbot: " + response)
    else :
        if st.button("Commencez l' enregistrement :"):
            text = transcribe_speech()
            st.write("Transcription: ", text)
if __name__ == "__main__":
    main()
