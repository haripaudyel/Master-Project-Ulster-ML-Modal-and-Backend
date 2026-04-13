import nltk
import random
import string

from spellchecker import SpellChecker
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Initialize NLP Tools
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
spell = SpellChecker()

# -----------------------------
# Spell Correction
# -----------------------------
def correct_spelling(text: str) -> str:
    corrected_words = []
    for word in text.split():
        if word.isalpha():
            corrected = spell.correction(word)
            if corrected is None:
                corrected_words.append(word)
            else:
                corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)

# -----------------------------
# Training Data
# -----------------------------
training_data = {
   "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "is anyone there",
        "can you help me",
        "i need help"
    ],

    "about_ulster": [
        "what is ulster university",
        "tell me about ulster",
        "about ulster university",
        "where is ulster university",
        "ulster university overview",
        "is ulster a good university"
    ],

    "campuses": [
        "ulster university campuses",
        "where are ulster campuses located",
        "belfast campus",
        "coleraine campus",
        "jordanstown campus",
        "magee campus",
        "how many campuses does ulster have"
    ],

    "admission_undergraduate": [
        "how can i apply for undergraduate",
        "ulster undergraduate admission process",
        "entry requirements for undergraduate",
        "how to get admission in ulster",
        "undergraduate application steps",
        "how to apply to ulster university"
    ],

    "admission_postgraduate": [
        "how to apply for masters at ulster",
        "ulster postgraduate admission",
        "masters admission requirements",
        "postgraduate application process",
        "msc admission ulster"
    ],

    "international_admission": [
        "international student admission ulster",
        "how can international students apply",
        "apply from india to ulster",
        "international entry requirements",
        "english requirements ulster",
        "ielts requirement for ulster"
    ],

    "courses": [
        "available courses at ulster",
        "data science course ulster",
        "ai course ulster",
        "computer science program ulster",
        "business management course ulster",
        "cyber security course ulster",
        "msc data science ulster",
        "bsc computer science ulster"
    ],

    "fees": [
        "what is the fee at ulster",
        "tuition fee ulster university",
        "ulster fee structure",
        "how much do i need to pay",
        "international student fees ulster",
        "masters fee at ulster"
    ],

    "scholarships": [
        "ulster scholarships",
        "international scholarships ulster",
        "scholarships for indian students",
        "financial aid ulster",
        "tuition fee discount ulster",
        "merit scholarship ulster"
    ],

    "deadlines": [
        "ulster application deadline",
        "when is the last date to apply",
        "masters application deadline ulster",
        "undergraduate deadline ulster",
        "intake deadlines ulster",
        "september intake ulster",
        "january intake ulster"
    ],

    "intakes": [
        "ulster intake months",
        "september intake ulster",
        "january intake ulster",
        "when does ulster intake start",
        "available intakes ulster"
    ],

    "accommodation": [
        "ulster accommodation",
        "student housing ulster",
        "hostel facilities ulster",
        "belfast campus accommodation",
        "how to apply for accommodation ulster"
    ],

    "visa": [
        "uk student visa for ulster",
        "visa process for ulster university",
        "how to get uk student visa",
        "cas letter ulster",
        "visa requirements for international students"
    ],

    "ranking": [
        "ulster university ranking",
        "is ulster ranked",
        "ulster university world ranking",
        "ulster ranking in uk",
        "ulster ranking for computer science"
    ],

    "placements": [
        "ulster placements",
        "job opportunities after ulster",
        "career support ulster",
        "graduate employability ulster",
        "placement rate ulster"
    ],

    "contact": [
        "contact details ulster",
        "ulster university email",
        "ulster phone number",
        "how to contact ulster university",
        "ulster international office contact"
    ],

    "location": [
        "where is ulster university located",
        "ulster university in which country",
        "ulster belfast location",
        "ulster campus locations"
    ],

    "fallback": [
        "sorry i didn't understand",
        "can you repeat",
        "i am not sure",
        "that doesn't help",
        "please explain again"
    ]
}

responses = {
    "greeting": [
        "Hello! Welcome to Ulster University support. How can I help you today?",
        "Hi there! Ask me anything about Ulster University."
    ],

    "about_ulster": [
        "Ulster University is a public university in Northern Ireland with campuses in Belfast, Coleraine and Derry~Londonderry (Magee).",
        "Ulster University offers career-focused programs with strong industry links and research excellence. It is known for its supportive learning environment and vibrant student community."
    ],

    "campuses": [
        "Ulster University has campuses in Belfast, Coleraine and Derry~Londonderry (Magee).",
        "Each Ulster campus specializes in different subject areas and facilities."
    ],

    "admission_undergraduate": [
        "You can apply for undergraduate courses at Ulster University through UCAS.",
        "Entry requirements vary by course. You should check academic and English language criteria for your chosen program."
    ],

    "admission_postgraduate": [
        "Postgraduate applications are submitted directly through the Ulster University online portal.",
        "You need a relevant bachelor's degree and proof of English proficiency for master's programs."
    ],

    "international_admission": [
        "International students can apply directly via Ulster University’s website.",
        "You will need academic transcripts, English test scores (IELTS), and a valid passport."
    ],

    "courses": [
        "Ulster University offers programs in Computer Science, Artificial Intelligence, Data Science, Business, Engineering, Health Sciences and more.",
        "You can browse undergraduate and postgraduate courses by subject area on Ulster’s website."
    ],

    "fees": [
        "Tuition fees depend on your course and whether you are a UK or international student.",
        "International fees vary by program. The latest fee details are available on Ulster University’s website."
    ],

    "scholarships": [
        "Ulster University offers international scholarships and tuition fee discounts based on merit.",
        "Scholarships vary by country and course. Check Ulster’s scholarships page for current offers. "
    ],

    "deadlines": [
        "Deadlines depend on the course and intake. Popular intakes are September and January.",
        "It’s recommended to apply early as some courses fill quickly. Check the specific course page for exact deadlines."
    ],

    "intakes": [
        "Ulster University mainly has September and January intakes for most courses.",
        "Not all courses are available in every intake. Check the specific course page for intake information."
    ],

    "accommodation": [
        "Ulster University offers student accommodation near its campuses.",
        "You can apply for university accommodation after receiving your offer letter. Options include halls of residence and private housing."
    ],

    "visa": [
        "International students need a UK Student Visa to study at Ulster University.",
        "After receiving an offer, Ulster will issue a CAS letter to support your visa application. You will also need to meet financial and English language requirements."
    ],

    "ranking": [
        "Ulster University is recognized in UK and global university rankings.",
        "Rankings vary by subject and year. Check the latest rankings on Ulster’s website or ranking platforms."
    ],

    "placements": [
        "Ulster University provides strong career support and placement opportunities.",
        "Many programs include industry placements or internships."
    ],

    "contact": [
        "You can contact Ulster University via the official website contact page or international office.",
        "Ulster’s admissions and international teams are available to help with applications."
    ],

    "location": [
        "Ulster University is located in Northern Ireland, UK.",
        "Its main campuses are in Belfast, Coleraine and Derry~Londonderry."
    ],

    "fallback": [
        "Sorry, I didn’t quite get that. Could you rephrase your question?",
        "I’m not sure about that yet. Try asking in a different way or check Ulster University’s website for more information."
    ]
}

# -----------------------------
# Prepare Training Dataset
# -----------------------------
sentences = []
labels = []

for intent, examples in training_data.items():
    for example in examples:
        corrected = correct_spelling(example)
        sentences.append(preprocess(corrected))
        labels.append(intent)

# -----------------------------
# Build ML Pipeline
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(sentences, labels)

# -----------------------------
# Get Chatbot Response
# -----------------------------
def get_response(user_input: str) -> str:
    corrected_input = correct_spelling(user_input)
    processed_input = preprocess(corrected_input)

    intent = model.predict([processed_input])[0]
    confidence = max(model.predict_proba([processed_input])[0])

    if confidence < 0.1:
        return random.choice(responses["fallback"])

    return random.choice(responses[intent])
