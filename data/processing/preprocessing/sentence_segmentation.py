from nltk.tokenize import sent_tokenize


# def sentence_segmentation(text): #A wrapper function that applies NLTK’s sent_tokenize on text
#     return sent_tokenize(text)
#
#
def sliding_window(text, window_size=512, overlap=128):
    words = text.split()
    segments = []

    for i in range(0, len(words), window_size - overlap):  # Shift by (window_size - overlap)
        segment = " ".join(words[i:i + window_size])
        segments.append(segment)

    return segments


def segment_text(text, window_size=512, overlap=128):
    words = text.split()

    # If the text is too short, use sentence segmentation instead
    if len(words) < window_size:
        return sent_tokenize(text)  # Use NLTK's sentence segmentation

    return sliding_window(text, window_size, overlap)