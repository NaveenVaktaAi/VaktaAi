import random

messages = [
    "You're welcome!",
    "Glad to help!",
    "Anytime!",
    "Happy to assist!",
    "No problem!",
    "My pleasure!",
    "Sure thing!",
    "Of course!",
    "Not a problem!",
    # "Certainly!",
    # "You bet!",
    "Always here to help!",
    # "It's what I'm here for!",
    "Absolutely!",
    # "Don't mention it!",
    "No worries!",
    "Always!",
    "With pleasure!",
    "Indeed!",
    "Glad I could assist!",
]


def create_thanking_response():
    random_index = random.randint(0, len(messages) - 1)

    return messages[random_index]
