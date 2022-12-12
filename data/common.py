import re

def processing_sequence(sequence: str):
    sequence = sequence.lower()
    re.sub(r'([?.!,¿@=+/#])', '', sequence)
    re.sub(r'\s\s+', ' ', sequence)
    sequence = sequence.strip()
    return sequence