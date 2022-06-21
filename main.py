import nltk


def read_file_to_str(directory, filenames):
    '''Opens a directory that contains text files and appends those to a list of strings'''
    strings = []
    for filename in filenames:
        with open(f'texts/{directory}/{filename}.txt', 'r') as f:
            strings.append(f.read())
    return '\n'.join(strings)


### Stylometry begins here ###

# download nltk's punkt for punctuation etc.
nltk.download('punkt')


def tokenize(text, language):
    '''Tokenises a given text (text) defined above and returns a list of tokens (tokens)'''
    tokens = nltk.word_tokenize(text=text.lower(), language=f"{language}")
    # strip punctuation of the list of word tokens:
    tokens = ([token for token in tokens if any(c.isalpha() for c in token)])
    return tokens


# tokens = tokenize(test, 'german')


def composition(tokens, title):
    '''creates a curve of composition for a corpus of texts'''
    token_lengths = [len(token) for token in tokens]
    len_distr = nltk.FreqDist(token_lengths)
    print(len_distr)
    len_distr.plot(15, title=f'{title}')


def run_all_comp(text, language, title):
    '''tokenizes a set of texts and creates a curve of composition for these texts'''
    tokenize(text, language)
    composition(tokenize(text, language), title=f"{title}")


# Create a string for all texts in the specified directories
german = read_file_to_str('german', filenames=['augsburger_rf'])
french_ue = read_file_to_str('french_ue', filenames=['amboise_ue'])
french_orig = read_file_to_str('french_orig', filenames=['amboise_orig'])

# Create the Curves of Composition for the German texts (Module 1) and the French texts(Module 5)
run_all_comp(german, 'german', title='german')
run_all_comp(french_ue, 'german', title='frz_ue')
run_all_comp(french_orig, 'french', title='frz_orig')


# Todo 2.2: Chi-Squared Method
def chi_squared(corpus_1, corpus_2, most_common_values):
    '''calculates chi-squared value for two combined corpora and identifies a stylistic similarity between them
    smaller value indicates a closer relationship.
    '''
    combined = (corpus_1 + corpus_2)
    combined_freq_dist = nltk.FreqDist(combined)
    most_common = list(combined_freq_dist.most_common(most_common_values))
    shared = len(corpus_1) / len(combined)
    chisquared = 0
    for word, joint_count in most_common:
        corpus_1_count = corpus_1.count(word)
        corpus_2_count = corpus_2.count(word)
        expected_corpus_1_count = joint_count * shared
        expected_corpus_2_count = joint_count * (1 - shared)
        chisquared += ((corpus_1_count - expected_corpus_1_count) * (
                corpus_1_count - expected_corpus_1_count) / expected_corpus_1_count)
        chisquared += ((corpus_2_count - expected_corpus_2_count) * (
                corpus_2_count - expected_corpus_2_count) / expected_corpus_2_count)

    print(f"The Chi-sqared value is {chisquared}")


chi_squared(german, french_ue, most_common_values=105)

chi_squared(french_orig, french_ue, most_common_values=105)
