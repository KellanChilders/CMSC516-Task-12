def get_ngrams(sentence, n):
    sentence = sentence.split()
    return [sentence[i:i+n] for i in range(len(sentence)+1 - n)]

if __name__ == "__main__":
    print(get_ngrams("This was a triumph", 1))
    print(get_ngrams("I'm writing a note here, huge success", 2))
    print(get_ngrams("It's hard to overstate my satisfaction", 3))
    # Sadly nltk does it just as well, so this will never get used.
