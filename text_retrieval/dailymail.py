
from bs4 import BeautifulSoup
import chardet
import glob


def retrieve_meta_data(soup):
    meta_content = dict()
    meta_info = {
                 'keywords':            {'s_type': "name", 's_prop': "keywords", 'target': 'keywords'},
                 'news_keywords':       {'s_type': "name", 's_prop': "news_keywords", 'target': 'keywords'},
                 'description':         {'s_type': "name", 's_prop': "description", 'target': 'description'},
                 'twitter_title':       {'s_type': "property", 's_prop': "twitter:title", 'target': 'twitter_title'},
                 'twitter_description': {'s_type': "property", 's_prop': "twitter:description", 'target': 'description'}
                 }

    for meta_key in meta_info:
        if 's_prop' in meta_info[meta_key]:
            s = soup.find("meta",  {meta_info[meta_key]['s_type']: meta_info[meta_key]['s_prop']})
            if s is not None:
                target = meta_info[meta_key]['target']
                if target not in meta_content: meta_content[target] = ""
                if s['content'] is not None and s['content'] > meta_content[target]:
                    meta_content[target] = s['content']

    return meta_content



path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/'
count = 0
for file_name in glob.iglob(path+"*.html"):
    #file_name = path + '0a0db6c641cedb28c98a2a8de080ec47b68bb9e7.html'
    char_type = chardet.detect(open(file_name, 'rb').read())['encoding']
    print('encoding', char_type, file_name)


    soup = BeautifulSoup(open(file_name, encoding=char_type).read()
                         .replace("font-size: 1", "font-size:1")
                         .replace(" font-size:1", "font-size:1")
                         .replace(';">', '">')
                         )

    meta_data = retrieve_meta_data(soup)

    soup_article = soup.find("div", id="js-article-text")
    excluded_terms = ['factbox', 'mol-bold']
    for element in [" ".join(c["class"]) for c in soup_article.find_all(class_=True)]:
        for term in excluded_terms:
            if term in element:
                for x in soup_article.findAll(class_=element): x.extract()

    #print(soup_article.prettify())


    headline = soup_article.find("h1").text.strip()

    summary_items = [[s.text for s in el.find_all("li")] for el in [soup_article.find("ul")]][0]

    #summary_items = [el.find("font", style='font-size:1.4em') for el in soup_article.find_all("li")]
    #summary_items = [el.text for el in summary_items if el is not None]
    #text_content = [el.find("font", style='font-size:1.2em') for el in soup_article.find_all("p")]
    #if len(text_content) == 0 or len([el for el in text_content if el is not None]) == 0:
        #text_content += [el.find("span", style='font-size:14px') for el in soup_article.find_all("p")]
    #text_content = [el.text.replace("\n", " ").strip() for el in text_content if el is not None and len(el.text) > 3]


    print(headline, "\n")
    print(meta_data['description'], "\n")
    for s in summary_items: print(s)
    print()
    '''
    for t in text_content: print(t)
    print("\n", "------------------------------------", "\n")
    '''
    print(summary_items)







