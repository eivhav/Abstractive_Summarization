
import glob, json, cchardet
from bs4 import BeautifulSoup
def retrieve_meta_data_cnn(soup):
    meta_content = dict()

    meta_info = {
        'description': {'s_type': "name", 's_prop': "description", 'target': 'description'},
        'timestamp_pub': {'s_type': "itemprop", 's_prop': 'dateCreated', 'target': 'timestamp_pub'},
        'headline': {'s_type': "itemprop", 's_prop': 'headline', 'target': 'headline'}
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

def parse_online_tokenized(text):

    def replace(s):
        return s.replace("`", "'").replace("``", "''").replace("-LRB-", "(").replace("-RRB-", ")")\
            .replace("   ", " ").replace("  "," ").strip()

    def fix_punct(s):
        if len(s) < 5: return s
        if s.strip()[-1] != "." and s.strip()[-1] != "!" and s.strip()[-1] != "?": return s.strip() + " ."
        return s.strip()

    def remove_NEW(s):
        if 'NEW :' == s[:5]: return s[5:].strip()
        return s


    story = ""
    story_and_highlights = text.split("\n\n@highlight")
    story_lines = [s.replace("\n", " ").strip() for s in story_and_highlights[0].split("\n\n")]
    story_lines = [s for s in story_lines if len(s) > 30]
    for line in story_lines:
        if "." not in line[-6:] and "!" not in line[-6:] and "?" not in line[-6:]:
            story = story + line.strip() + " . "
        else:
            story = story + line.strip() + " "
    story = replace(story)
    summary = ""
    if len(story_and_highlights) > 1:
        summary = [fix_punct(remove_NEW(h.replace("\n", "").strip())) for h in story_and_highlights[1:] if len(h) > 5]

    return {'online_summary': summary, 'online_text_content': story}


def correct_online(data_path, tok_folder, online_folder,):

    tokenized_data = dict()
    print("loading tokenized data")
    files = list(glob.iglob(data_path + tok_folder + "*.txt"))

    for file in files:
        d = json.load(open(file))
        for k in d.keys(): tokenized_data[k.split("/")[-1].split(".html")[0]] = d[k]

    online_tok_data = dict()
    files = list(glob.iglob(data_path + online_folder + "*.story"))
    print("loading parsing data data")
    count = 0

    for file in files:
        if file.split("/")[-1].split(".story")[0]:
            count += 1
            text = open(file).read()
            online_tok_data[file.split("/")[-1].split(".story")[0]] = parse_online_tokenized(text)
            if count % 1000 == 0: print(count)

    print("merging data")
    count = 0
    results = dict()
    for key in tokenized_data:
        if key in online_tok_data and len(online_tok_data[key]['online_text_content']) > 40:
            for k in online_tok_data[key]:
                tokenized_data[key][k] = online_tok_data[key][k]
            count += 1
            results[key] =tokenized_data[key]
            if count % 1000 == 0:
                with open(data_path+ "combined_3/" + "DM_" + str(int(count / 1000)) + "000" + ".txt", "w") as handle:
                   handle.write(json.dumps(results))
                results = dict()
            if count % 1000 == 0: print(count)

    with open(data_path + "combined_3/" + "DM_last" + ".txt", "w") as handle:
        handle.write(json.dumps(results))
    results = dict()


data_path = '/home/havikbot/MasterThesis/Data/DailyMail/'
tok_folder = 'combined/'
online_folder = 'dm_tokenized_online/'

correct_online(data_path, tok_folder, online_folder)






def join_cnn(data_path, html_folder, tok_folder, online_folder):

    files = list(glob.iglob(data_path + "combined/" + "*.txt"))
    exclude = dict()
    for file in files:
        d = json.load(open(file))
        for k in d.keys(): exclude[k] = True

    html_data = dict()
    '''
    files = list(glob.iglob(data_path + html_folder + "*.html"))
    print("Soup'ing html")
    count = 0
    for file in files:
        if file.split("/")[-1].split(".html")[0] not in exclude:
            try:
                count += 1
                char_type = cchardet.detect(open(file, 'rb').read())['encoding']
                soup = BeautifulSoup(open(file, encoding=char_type).read(), "lxml")
                html_data[file.split("/")[-1].split(".html")[0]] = retrieve_meta_data_cnn(soup)
                if count % 10 == 0: print(count)
            except:
                print("failed for encoding")
    '''
    tokenized_data = dict()
    print("loading tokenized data")
    files = list(glob.iglob(data_path + tok_folder + "*.txt"))
    for file in files:
        d = json.load(open(file))
        for k in d.keys(): tokenized_data[k.split("/")[-1].split(".html")[0]] = d[k]

    online_tok_data = dict()
    files = list(glob.iglob(data_path + online_folder + "*.story"))
    print("loading parsing data data")
    count = 0
    for file in files:
        if file.split("/")[-1].split(".story")[0] not in exclude:
            count += 1
            text = open(file).read()
            d = parse_online_tokenized(text)

            online_tok_data[file.split("/")[-1].split(".story")[0]] = d
            html_data[file.split("/")[-1].split(".story")[0]] = {}
            if count % 1000 == 0: print(count)

    print("merging data")
    count = 0
    results = dict()
    for key in online_tok_data:
        if key in tokenized_data and key in html_data and len(online_tok_data[key]['online_text_content']) > 40:
            for k in tokenized_data[key]:
                online_tok_data[key][k] = tokenized_data[key][k]
            for k in html_data[key]:
                online_tok_data[key][k] = html_data[key][k]
            count += 1
            results[key] = online_tok_data[key]
            if count % 1000 == 0:
                with open(data_path+ "combined/" + "DM_" + str(int(count / 1000)) + "000" + ".txt", "w") as handle:
                   handle.write(json.dumps(results))
                results = dict()
            if count % 1000 == 0: print(count)

    with open(data_path + "combined/" + "DM_last" + str(int(count / 1000)) + "000" + ".txt", "w") as handle:
        handle.write(json.dumps(results))
    results = dict()





html_folder = 'downloads/'
data_path = '/home/havikbot/MasterThesis/Data/DailyMail/'
tok_folder = 'tokenized_common/'
online_folder = 'dm_tokenized_online/'


#join_cnn(data_path, html_folder, tok_folder, online_folder)























