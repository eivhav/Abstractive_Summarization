
from bs4 import BeautifulSoup
import cchardet as cchardet
import glob
import json
from multiprocessing import Process


def retrieve_meta_data(soup):
    meta_content = dict()
    meta_info = {
                 'keywords':            {'s_type': "name", 's_prop': "keywords", 'target': 'keywords'},
                 'news_keywords':       {'s_type': "name", 's_prop': "news_keywords", 'target': 'keywords'},
                 'description':         {'s_type': "name", 's_prop': "description", 'target': 'description'},
                 'twitter_title':       {'s_type': "property", 's_prop': "twitter:title", 'target': 'twitter_title'},
                 'twitter_description': {'s_type': "property", 's_prop': "twitter:description", 'target': 'description'},
                 'timestamp_pub':       {'s_type': "property", 's_prop': 'article:published_time', 'target': 'timestamp_pub'}
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



def decode_and_write_dm_to_json(file_names, path, out_path, p_id):
    count = 0
    data = dict()
    for file_name in file_names:
        count += 1
        if count % 1000 == 0:
            with open(out_path+"DM_"+str(int(count/1000)) + "000_"+str(p_id)+".txt", "w") as handle:
                handle.write(json.dumps(data))
            data = dict()
        if count % 100 == 0: print(p_id, count)

        try:
            char_type = cchardet.detect(open(file_name, 'rb').read())['encoding']
            soup = BeautifulSoup(open(file_name, encoding=char_type).read()
                                 .replace("font-size: 1", "font-size:1")
                                 .replace(" font-size:1", "font-size:1")
                                 .replace(';">', '">')
                                 .replace('font-size:14px', 'font-size:1.2em')
                                 )

            meta_data = retrieve_meta_data(soup)

            soup_article = soup.find("div", id="js-article-text")
            excluded_terms = ['factbox', 'mol-bold', 'imageCaption', 'article-icon', 'article-timestamp', 'art-ins', 'floatRHS']
            for element in [" ".join(c["class"]) for c in soup_article.find_all(class_=True)]:
                for term in excluded_terms:
                    if term in element:
                        for x in soup_article.findAll(class_=element): x.extract()
            #print(soup_article.prettify())
            headline = soup_article.find("h1").text.strip()
            summary_items = [[s.text.replace("\n", " ") for s in el.find_all("li")] for el in [soup_article.find("ul")]][0]

            for id in ["font", "span", None]:
                if id is None: text_content = [[el] for el in soup_article.find_all("p")]
                else: text_content = [el.find_all(id, style='font-size:1.2em') for el in soup_article.find_all("p")]
                text_content = [" ".join([e.text.replace("\n", " ").strip() for e in el if e is not None]) for el in text_content if len(el)>0]
                text_content = [t.replace("  ", " ") for t in text_content if len(t)> 0 and not t[-1].isalpha()]
                if len(text_content) > 3: break

            meta_data['encoding'] = char_type
            meta_data['headline'] = headline
            meta_data['summary_items'] = summary_items
            meta_data['text_content'] = text_content
            data[file_name[len(path):]] = meta_data


            '''
            print(meta_data['timestamp_pub'])
            print(headline, "\n")
            print(meta_data['description'], "\n")
            for s in summary_items: print("*", s)
            print()
            for t in text_content: print(t)

            print("\n", "------------------------------------", "\n")
            '''
        except:
            print("Error with decoding file")


if __name__ == '__main__':
    path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/'
    out_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/extracted/'
    path_toshiba = '/media/havikbot/TOSHIBA EXT/MasterThesis/Data/dailymail/downloads/'
    file_names = list(glob.iglob(path_toshiba+"*.html"))

    nb_processes = 6
    tasks = []
    for i in range(nb_processes): tasks.append([])
    processes = []
    for i in range(len(file_names)): tasks[i % nb_processes].append(file_names[i])
    for p in range(nb_processes):
        proc = Process(target=decode_and_write_dm_to_json, args=(tasks[p], path, out_path, p))
        proc.start()
        processes.append(proc)
    for p in processes: p.join()















