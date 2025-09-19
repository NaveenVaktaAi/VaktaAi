from enum import Enum

import tiktoken


class EncodingName(Enum):
    """
    Encoding names:
    text-embedding-ada-002 = cl100k_base
    code models, text-davinci-002, text-davinci-003 = p50k_base
    rest of GPT models = r50k_base
    """

    cl100k_base = "cl100k_base"
    p50k_base = "p50k_base"
    r50k_base = "r50k_base"

    EMBEDDING_ADA_2 = "cl100k_base"
    TEXT_DAVINCI_2 = "p50k_base"
    TEXT_DAVINCI_3 = "p50k_base"
    GPT_3_5_TURBO = "cl100k_base"
    GPT_4 = "cl100k_base"
    GPT_4_32k = "cl100k_base"


_encodings = {}

for _encoding_name in EncodingName:
    _encodings[_encoding_name] = tiktoken.get_encoding(_encoding_name.value)


def num_tokens_from_string(
    string: str | list[dict[str, str]], encoding_name: EncodingName
) -> int:
    """
    Returns the number of tokens in a text string.
    """

    encoding = _encodings[encoding_name]
    if isinstance(string, list):
        num_tokens = 0
        for message in string:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        num_tokens = len(encoding.encode(string))
    return num_tokens


def get_encoding(encoding_name: EncodingName):
    return _encodings[encoding_name]
