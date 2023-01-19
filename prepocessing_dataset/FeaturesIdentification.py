import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from geotext import GeoText
from nltk.tokenize import sent_tokenize
import re
import string
import spacy
# # import en_core_web_sm
# # nlp = en_core_web_sm.load()
# from spacy.lang.en.examples import sentences
# nlp = spacy.load('en_core_web_sm')
# nlseg = NewLineSegmenter()
# nlp.add_pipe(nlseg.set_sent_starts, name='sentence_segmenter', before='parser')
stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
interjections_following_1 = ["a", "so", "very", "for", "in"]
interjections_following_2 = ["lot", "much", "million", "advance", "all", "everything", "your", "the"]


#
# Method: check the presence of imperative question, e.i. imperative verbs at the beginning of the sentence, from a list of verbs
#
def sentence_detection(sentences):
    for sentence in sentences:
        tokens = nltk.word_tokenize(str(sentence))
        if tokens[0].lower() in imperative_lexicon:
            if tokens[0].lower() != "please":
                    return 1
            else:
                if not tokens[-1].lower():
                    return 1

    return 0


#
# Method: check the presence of sensitive data as time, dates, fiscal code, email adresses
#
def sensitive_detection(text):
    if (bool(re.search('(\d{1,2})[.:](\d{1,2})?([ ]?(am|pm|AM|PM))?', text)) == True)or (bool(re.search("^(\\d{1,}) [a-zA-Z0-9\\s]+(\\,)? [a-zA-Z]+(\\,)? [A-Z]{2} [0-9]{5,6}$", text)) == True) or (bool(re.search(r'\w*\d\w*', text)) == True) or (bool(re.search("\d{2}[- /.]\d{2}[- /.]\d{,4}", text)) == True) or (bool(re.search("(([a-zA-Z0-9_+]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))", text)) == True):
        return 1
    return 0


def check_for_complex_thank(token_1, token_2):
    if token_1 in interjections_following_1:
        if token_2 in interjections_following_2:
            return 1
    return 0


if __name__ == '__main__':
    # read all json file
    list_of_files = sorted(glob.glob('CSV_train/*.csv'))
    imperative_lexicon_pd = pd.read_csv("imperativeVB_lexicon_complete.csv", encoding='utf-8', delimiter=';')
    print(">>> Export imperative lexicon to dict...")
    imperative_lexicon = {}
    for index, row in imperative_lexicon_pd.iterrows():
        imperative_lexicon[row["word"]] = row["freq"]

    names_lexicon_pd = pd.read_csv("babynames-clean.csv", encoding='utf-8', delimiter=';')
    print(">>> Export names lexicon to dict...")
    firstnames_lexicon = {}
    for index, row in names_lexicon_pd.iterrows():
        firstnames_lexicon[row["Name"]] = row["Gener"]


    for file_name in list_of_files:
        print(">>> Filename: " + file_name)
        df = pd.read_csv(file_name, encoding='utf-8', delimiter=',')
        df_features = pd.DataFrame(columns=["dialogue_id", "turn", "words", "sentences", "quest_mark", "whq", "imper_quest", "places_services", "simple_quest", "sensitive_data", "interjections", "conditional_vb"])

        wh_q = ["what", "why", "who", "where", "when", "how", "whose"]
        interjections = ["please", "great", "sorry"]
        conditional_verbs = ["'d", "would", "wouldn", "should", "shouldn", "could", "couldn", "might"]

        print(">>> Check for features...")

        id = ""
        for index, row in df.iterrows():
            quest_mark = 0
            wh = 0
            imperative_q = 0
            places = 0
            simple_quest = 0
            sensitive_data = 0
            inter = 0
            conditional_vb = 0
            if row["speaker"] == "USER":
                tokens = nltk.word_tokenize(row['turn'])
                # tokens = [w for w in tokens if not w.lower() in stop_words]
                words = len([i for i in tokens if i not in punctuations])
                # if ? is present, there is a question
                if "?" in row['turn']:
                    quest_mark = 1
                # if places names are present, there is a use of specific name
                if GeoText(row['turn']).cities or GeoText(row['turn']).countries or GeoText(
                        row['turn']).country_mentions or GeoText(row['turn']).nationalities:
                    places = 1
                for token in tokens:
                    # if there is ? and a wh, there is a wh-question
                    if token.lower() in wh_q and quest_mark == 1:
                        wh = 1
                    # if there is a first name, sensitive data are used (excluding name of places and names similar to verbs)
                    if token in firstnames_lexicon and places == 0 and token != "May" and token != "Will":
                        sensitive_data = 1

                    # New features #
                    if token.lower() in interjections:
                        inter = 1
                    # if thanks is not the last word...
                    if token.lower() == "thank" or token.lower() == "thanks":
                        if tokens.index(token) != len(tokens)-1:
                            next_token = tokens[tokens.index(token)+1]
                            if tokens.index(next_token) != len(tokens)-1:
                                if next_token.lower() == "you":
                                    next_token_1 = tokens[tokens.index(next_token)+1]
                                    if tokens.index(next_token_1) != len(tokens)-1:
                                        next_token_2 = tokens[tokens.index(next_token_1) + 1]
                                        inter=check_for_complex_thank(next_token_1.lower() , next_token_2.lower())
                                else:
                                    next_token_1 = tokens[tokens.index(next_token) + 1]
                                    if tokens.index(next_token_1) != len(tokens)-1:
                                        next_token_2 = tokens[tokens.index(next_token_1) + 1]
                                        inter=check_for_complex_thank(next_token_1.lower() , next_token_2.lower())
                    # END #
                    if token.lower() in conditional_verbs:
                        conditional_vb = 1
                sentences = sent_tokenize(row['turn']) # dividing sentences to check imperative presence (utt beginning)
                imperative_q = sentence_detection(sentences)
                sensitive_data = sensitive_detection(row['turn'])
                # if there are less then 6 words and a ?, it is considered a simple question, but there is no reason behind number 6 (it can be changed)
                if words <= 6 and quest_mark == 1:
                    simple_quest = 1


                df_features.loc[-1] = [row['dialogue_id'], row['turn'], words, len(sentences), quest_mark, wh, imperative_q,
                                       places, simple_quest, sensitive_data, inter, conditional_vb]
                df_features.index = df_features.index + 1

        n_file = file_name[23:26]
        df_features.to_csv("features/utterances_features_" + n_file + ".csv", index=False)