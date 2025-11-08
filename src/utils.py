import re

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def write_text_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')
    return file_path

def clean_entities(text):
    # Remove any content in parentheses, along with any leading space
    text = re.sub(r'\s*\([^)]*\)', '', text)
    # Remove any hashtag content (hash and following non-space characters), along with any leading space
    text = re.sub(r'\s*#[^\s]*', '', text)
    return text