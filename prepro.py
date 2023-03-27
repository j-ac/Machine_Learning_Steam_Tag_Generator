import csv
import nltk
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#filename = "first30.csv"
filename = "Steam_games3.csv"
#filename = "test.csv"

fields = []
rows = []

VERBOSE = True
MINIMUM_DESC_LENGTH = 100   # Discard any game in the initial pass with a description less than this this (in characters)
MINIMUM_NUM_TAGS = 5        # Because lots of bad data has few tags. Similar manner to above
IS_USING_ID_OUTPUT = True  # If True then print id numbers (0,1,2...n) otherwise print the words themselves
MINIMUM_FREQUENCY = 100     # Don't include words that don't appear in at least this many games
MAXIMUM_FREQUENCY = 5000    # Same as above but as a max

class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens

# Given a dictionary relating an index to a word list (ie a description or tag list) return a new list eliminating keys outside the range
def filter_dict_by_frequency(minimum, maximum, dct):
    # Calculate the frequency of each word
    word_frequencies = {} # word -> frequency
    for word_list in dct.values():
        for word in word_list:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] = word_frequencies[word] + 1


    filtered = {}
    for word_list in dct.items():
        fltr = list(filter(lambda word: word_frequencies[word] >= minimum and word_frequencies[word] <= maximum, word_list[1]))
        fltr = list(filter(lambda word: word.isalpha(), fltr))
        filtered[word_list[0]] = fltr # TODO word_list[0] just includes an index 0,1,2....n. Definitely wasteful instead of using enumerate()
                
    return filtered

# Given a dictionary of words, return a list of lists where the words for each description are converted into unique IDs
def convert_words_to_ids(dct):
    ids = {}
    i = 0

    # assign IDs
    for description in dct.values():
        for word in description:
            if word not in ids:
                ids[word] = i #associate this word with 1,2,3....
                i += 1

    # replace
    ret = []
    for description in dct.values():
        nums = []
        for word in description:
            nums.append(ids[word])

        ret.append(nums)
                       

    return ret

# Used to accomplish the list of list format desired by the csv converter when not getting id numbers (as convert_words_to_ids normally does this)
def convert_words_to_list_of_list(dct):
    ret = []
    for description in dct.values():
        words = []
        for wrd in description:
            words.append(wrd)

        ret.append(words)
        
    return ret
# Given two columns, make csv files
# Writes training.csv, dev.csv and test.csv. Divides the data among these according to the ratios specified in the float values.
def make_csv_files(col1, col2, fields, training_set: float, dev_set: float):
    zipped = zip(col1, col2)

    with open("training.csv", 'w') as csvtrain:
        trainwriter = csv.writer(csvtrain)
        trainwriter.writerow(fields)

        with open("dev.csv", 'w') as csvdev:
            devwriter = csv.writer(csvdev)
            devwriter.writerow(fields)

            with open("test.csv", 'w') as csvtest:
                testwriter = csv.writer(csvtest)
                testwriter.writerow(fields)

                for tup in zipped:
                    r = random.random()
                    if r < training_set:
                        trainwriter.writerow(tup)
                    elif r > training_set and r < training_set + dev_set:
                        devwriter.writerow(tup)
                    else:
                        testwriter.writerow(tup)


def main():
    # LOAD THE CSV
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # Read first row where fields are defined
        fields = next(csvreader)

        for row in csvreader:
            num_tags = len(row[0].split(","))
            desc_length = len(row[1])
            if (desc_length >= MINIMUM_DESC_LENGTH and num_tags >= MINIMUM_NUM_TAGS):
                rows.append(row)


    # PUT LEMMATIZED DESCRIPTIONS AND NON-LEMMATIZED TAGS INTO A DICTIONARY
    game_descs = {} # dictionary storing game's position in the list -> lemmatized description
    game_tags = {} # position in the list -> tag list (NOT LEMMATIZED)
    for i, game in enumerate(rows):
        # TAGS
        tags = game[0].split(",")
        game_tags[i] = tags

        # DESCRIPTIONS
        tokens = splitter.split(game[1])
        lowercase = [list(map(str.casefold, x)) for x in tokens]
        lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(lowercase)
 
        lemmatized = []
        for sentence in lemma_pos_token:
            for word in range(len(sentence)):
                lemmatized.append(sentence[word][1]) #extract just the lemma, ignoring the original word and the Part of Speech

        game_descs[i] = list(dict.fromkeys(lemmatized))
    
        if(VERBOSE and i % 2500 == 0):
            print("Initial pass: processed CSV line ", i)

    # FILTER THE DICTIONARIES
    filtered_descs = filter_dict_by_frequency(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY, dct=game_descs)
    filtered_tags = filter_dict_by_frequency(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY, dct=game_tags)

    if (IS_USING_ID_OUTPUT):
        words_ids = convert_words_to_ids(filtered_descs)
        tags_ids = convert_words_to_ids(filtered_tags)
    else:
        words_ids = convert_words_to_list_of_list(filtered_descs)
        tags_ids = convert_words_to_list_of_list(filtered_tags)
        
    make_csv_files(tags_ids, words_ids, ["tags", "desc"], 0.0, 0.0)

class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg   
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens


lemmatizer = WordNetLlemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

if (__name__ == "__main__"):
    main()
