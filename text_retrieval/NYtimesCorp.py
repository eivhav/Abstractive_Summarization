

import gzip

path = '/media/havikbot/TOSHIBA EXT/NYTcorpus/data/2007/01.tgz'

articles = []

with gzip.open(path,'r') as fin:
    lines = fin.readlines()
    article = ""
    for line in lines:
        utf_line = line.decode("utf-8")
        if utf_line[:len('<!DOCTYPE nitf SYSTEM')] == '<!DOCTYPE nitf SYSTEM':
            articles.append(article)
            article = ""
        else: article = article + utf_line

    if article != "": articles.append(article)



