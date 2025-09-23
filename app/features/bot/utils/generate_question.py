from app.features.bot.utils.response import ResponseCreator
# from question_generation import pipeline
from sentry_sdk import capture_exception


def break_paragraphs(paragraph):
    try:
        sentences = paragraph.split(".")
        result = []
        num_fullstops = 1

        while num_fullstops <= len(sentences):
            result.append(".".join(sentences[:num_fullstops]))
            num_fullstops *= 2

        for i in range(1, len(sentences) + 1):
            result.append(".".join(sentences[:i]))

        return list(set(result))
    except Exception as e:
        capture_exception(e)
        return []  # or handle the error in an appropriate way


# def generate_questions(story):
#     # Load the question-generation pipeline
#     nlp = pipeline("e2e-qg")
#     return nlp(story)


def generate_follow_up_questions(question, answer):
    more_questions = ResponseCreator().generate_questions_from_chatgpt(question)
    return [(f"{question}", answer) for question in more_questions]


def add_follow_up_questions(entry):
    # Generate follow-up questions for the current entry
    question = entry["User"]
    answer = entry["AI"]
    follow_ups = generate_follow_up_questions(question, answer)

    # Check if there are any 'Sub' keys
    has_sub_keys = any(key.startswith("Sub") for key in entry.keys())

    if has_sub_keys:
        # If 'Sub' keys exist, process them recursively
        for key in entry.keys():
            if key.startswith("Sub") and isinstance(entry[key], list):
                for sub_entry in entry[key]:
                    add_follow_up_questions(sub_entry)

                # Add follow-up questions to each 'Sub' list
                entry[key].extend([{"New User": q, "AI": a} for q, a in follow_ups])
    else:
        # If no 'Sub' keys, create one and add follow-up questions
        entry["Sub user"] = [{"New User": q, "AI": a} for q, a in follow_ups]
