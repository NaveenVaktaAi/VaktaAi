import random
 
acknowledgment_messages = [
    # "Got it!",
    # "Noted!",
    # "Understood!",
    # "Will do!",
    # "Confirmed!",
    # "Done!",
    # "On it!",
    # "Sounds good!",
    # "Perfect!",
    # "Got it, thanks!",
    # "Acknowledged!",
    # "Clear!",
    # "Sure thing!",
    # "Agreed!",
    # "Copy that!",
    # "Yep, done!",
    # "All set!",
    # "Updated!",
    # "Saved!",
    # "Got your point!",
    # "Alright!",
    # "Okay!",
    # "Sure!",
    # "Makes sense!",
    # "I see!",
    # "Works for me!",
    # "Fair enough!",
    # "Noted, thanks!",
    # "Okay, got it!",
    # "Right on!",
    # "Yep!",
    # "Cool!",
    # "Gotcha!",
    # "All good!",
    # "Okay, will do!",
    # "Understood, appreciate it!",

    "Alright!",
    "Got it!",
    "Perfect!",
    "Understood, thanks!",
    "Acknowledged",
    "Sounds good!",
    "Glad we're on the same page!",
    "You're welcome!",
    "Anytime!",
    "My pleasure",
    "Happy to help!",
    "No problem"
]
 
 
def create_confirmation_response():
    random_index = random.randint(0, len(acknowledgment_messages) - 1)
 
    return acknowledgment_messages[random_index]
 