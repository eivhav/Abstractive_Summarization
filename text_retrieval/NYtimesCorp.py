

import gzip
from bs4 import BeautifulSoup
import glob
import json
from multiprocessing import Process



def retrieve_data(soup):
    article = dict()
    article['timestamp_pub'] = soup.pubdata.attrs['date.publication']
    article_id = soup.pubdata.attrs['ex-ref']
    s_items_exclude = ['photos (M)']
    if soup.abstract:
        article['summary_items'] = [s for s in soup.abstract.text.strip().split(";") if s.strip() not in s_items_exclude]
    if soup.hedline:
        article['headline'] = [h.text.strip() for h in soup.hedline.find_all()]
    if soup.find('block', class_="lead_paragraph"):
        article['article_lead'] = [p.text.strip() for p in soup.find('block', class_="lead_paragraph").find_all('p')]
    if soup.find('block', class_="online_lead_paragraph"):
        article['description'] = [p.text.strip() for p in soup.find('block', class_="online_lead_paragraph").find_all('p')]
    if soup.find('block', class_="full_text"):
        article['text_content'] = " ".join([p.text.strip() for p in soup.find('block', class_="full_text").find_all('p')])
    return article, article_id

def read_zip_file(path, p):
    all_articles = dict()
    articles_with_abstract = dict()
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
    count = 0
    for a_html in articles[1:]:
        article, id = retrieve_data(BeautifulSoup(a_html))
        if 'summary_items' and 'text_content' in article:
            articles_with_abstract[id] = article
        if 'text_content' in article: all_articles[id] = article
        if count % 1000 == 0: print(p, count)
        count += 1

    return all_articles, articles_with_abstract


def read_parse_and_save_zip_data(years, data_path, out_path, p):
    print("process p", p, " years", years)
    for year in years:
        file_names = list(glob.iglob(data_path+str(year)+"/"+"*.tgz"))
        for file in file_names:
            print(file)
            all_articles, articles_with_abstract = read_zip_file(file, p)
            with open(out_path[0]+"NYT_all_"+str(year)+"_"+ file.split("/")[-1].replace(".tgz", "") + ".txt", "w") as handle:
                handle.write(json.dumps(all_articles))
            with open(out_path[1]+"NYT_abs_"+str(year)+"_"+ file.split("/")[-1].replace(".tgz", "") + ".txt", "w") as handle:
                handle.write(json.dumps(articles_with_abstract))






if __name__ == '__main__':
    data_path = '/home/havikbot/NYTcorpus/data/'
    out_path_abs = '/home/havikbot/MasterThesis/Data/NYTcorpus/with_abstract/json/'
    out_path_all = '/home/havikbot/MasterThesis/Data/NYTcorpus/all_articles/json/'

    nb_processes = 4
    tasks = []
    for p in range(nb_processes): tasks.append([])
    processes = []
    for year in range(1996, 2008): tasks[year % nb_processes].append(year)
    for p in range(nb_processes):
        proc = Process(target=read_parse_and_save_zip_data, args=(tasks[p], data_path, (out_path_all, out_path_abs), p))
        proc.start()
        processes.append(proc)
    for p in processes: p.join()





















