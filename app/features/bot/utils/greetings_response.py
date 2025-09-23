import random

messages = [
    "Greetings, what assistance may I provide?",
    "Hello there, how can I be of service?",
    "Hi, what can I assist you with?",
    "Hey, how may I assist you today?",
    "Good day! How can I be of help?",
    "Welcome! What do you need assistance with?",
    "Salutations! How may I aid you?",
    "Hello, how can I lend a hand?",
    "Hey there! What assistance do you require?",
    "Greetings! How may I be of service?",
    "Hello, how can I assist you today?",
    "Hi, what can I help you with?",
    "Greetings! What do you need help with?",
    "Hey there, what assistance can I offer?",
    "Hi, how can I assist you?",
    "Hello! How may I help you?",
    "Hi there! What do you need support with?",
    "Greetings, what can I do for you?",
    "Hey, how can I help?",
    "Hello there! What assistance do you need?",
    "How can I help?",
    "What's up?",
    "What can I do for you?",
]


def create_greetings_response():
    random_index = random.randint(0, len(messages) - 1)

    return messages[random_index]
