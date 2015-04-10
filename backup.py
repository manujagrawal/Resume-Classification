from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import fileinput
import nltk
from nltk.corpus import stopwords

stop = stopwords.words('english')
stop_words = {}
for i in range(len(stop)):
    stop_words[str(stop[i])] = True
sw = file('sample.txt', 'r')
for word in sw:
    if not stop_words.get(word, False):
        stop_words[word] = True

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
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text.lower()

def remove_stop_words(text, stop):
    text = text.split()
    modtext = []
    for i in range(len(text)):
        try:
            modtext.append(text[i].encode('ascii'))
        except UnicodeDecodeError:
            pass
    for word in modtext:
        if stop.get(word, False) or len(word)==1:
            modtext.remove(word)
    print(' '.join(modtext), len(modtext))

if __name__ == "__main__":
    try:
        text = convert_pdf_to_txt('resume1.pdf')
        remove_stop_words(text, stop_words)
    except IOError:
        pass