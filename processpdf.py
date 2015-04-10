from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import fileinput, string, pickle, os
import nltk, numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pandas as pd

stop = stopwords.words('english')
stop_words = {}
for i in range(len(stop)):
    stop_words[str(stop[i]).strip().encode('ascii')] = True
sw = file('sample.txt', 'r')
for word in sw:
    word = word.strip().strip('\n').encode('ascii')
    if not stop_words.get(word, False):
        stop_words[word] = True
sw.close()
global_word_count = {}

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    modtext = []
    for i in range(len(text)):
        try:
            text[i] = text[i].encode('ascii').strip()
            if not stop_words.get(text[i], False) or len(word) == 1:
                modtext.append(text[i])
        except UnicodeDecodeError:
            pass
    return ' '.join(modtext)

if __name__ == "__main__":
    try:
        train = pd.read_csv("CVs/cs/cs", header=0, delimiter=",")
        print(train.shape)
        cvs_text = []
        pdfs = os.listdir('CVs/cs')
        for pdf in pdfs:
            if pdf.endswith('pdf'):
                text = convert_pdf_to_txt('CVs/cs/'+pdf)
                processed_text = remove_stop_words(text)
                cvs_text.append(processed_text)
        # text = convert_pdf_to_txt('resume.pdf')
        # processed_text = remove_stop_words(text)
        # cvs_text.append(processed_text)
        vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words='english', \
                             max_features=5000)
        train_data_features = vectorizer.fit_transform(cvs_text)
        train_data_features = train_data_features.toarray()
        vocab = vectorizer.get_feature_names()
        dist = np.sum(train_data_features, axis=0)
        for tag, count in zip(vocab, dist):
            global_word_count[tag] = count
        #print(sorted(global_word_count.items(), key=lambda x:x[1], reverse=True))
        forest = RandomForestClassifier(n_estimators = 100)
        forest = forest.fit(train_data_features, train["vector"])
        # joblib.dump(forest, "random.pkl")
        # if len(sys.argv) > 1 and sys.argv[1] == "train":
        #     forest_predictor = joblib.load("random.pkl")
        # if len(sys.argv) > 1 and sys.argv[1] == "demo":
        #     forest_predictor = joblib.load("random.pkl")
    except IOError:
        pass