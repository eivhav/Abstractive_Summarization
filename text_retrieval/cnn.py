
import cchardet as chardet
from lxml import html

from itertools import chain
from collections import namedtuple
import types



class Story(namedtuple('StoryBase', 'url content highlights')):

  def ToString(self):
    return self.content + ''.join([
        '\n\n@highlight\n\n' + highlight
        for highlight in
        self.highlights])


AnonymizedStory = namedtuple('AnonymizedStory', 'url content highlights anonymization_info')
RawStory = namedtuple('RawStory', 'url html')
TokenizedStory = namedtuple('TokenizedStory', 'url tokens')




def ParseHtml(story, corpus):
  """Parses the HTML of a news story.
  Args:
    story: The raw Story to be parsed.
    corpus: Either 'cnn' or 'dailymail'.
  Returns:
    A Story containing URL, paragraphs and highlights.
  """

  parser = html.HTMLParser(encoding=chardet.detect(story.html)['encoding'])
  tree = html.document_fromstring(story.html, parser=parser)

  # Elements to delete.
  delete_selectors = {
      'cnn': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ],
      'dailymail': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ]
  }

  # Paragraph exclusions: ads, links, bylines, comments
  cnn_exclude = (
      'not(ancestor::*[contains(@class, "metadata")])'
      ' and not(ancestor::*[contains(@class, "pullquote")])'
      ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
      ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
      ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
      ' and not(contains(@class, "cnnTopics"))'
      ' and not(descendant::*[starts-with(text(), "Read:")])'
      ' and not(descendant::*[starts-with(text(), "READ:")])'
      ' and not(descendant::*[starts-with(text(), "Join us at")])'
      ' and not(descendant::*[starts-with(text(), "Join us on")])'
      ' and not(descendant::*[starts-with(text(), "Read CNNOpinion")])'
      ' and not(descendant::*[contains(text(), "@CNNOpinion")])'
      ' and not(descendant-or-self::*[starts-with(text(), "Follow us")])'
      ' and not(descendant::*[starts-with(text(), "MORE:")])'
      ' and not(descendant::*[starts-with(text(), "SPOILER ALERT:")])')

  dm_exclude = (
      'not(ancestor::*[contains(@id,"reader-comments")])'
      ' and not(contains(@class, "byline-plain"))'
      ' and not(contains(@class, "byline-section"))'
      ' and not(contains(@class, "count-number"))'
      ' and not(contains(@class, "count-text"))'
      ' and not(contains(@class, "video-item-title"))'
      ' and not(ancestor::*[contains(@class, "column-content")])'
      ' and not(ancestor::iframe)')

  paragraph_selectors = {
      'cnn': [
          '//div[contains(@class, "cnnContentContainer")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "l-container")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "cnn_strycntntlft")]//p[%s]' % cnn_exclude
      ],
      'dailymail': [
          '//div[contains(@class, "article-text")]//p[%s]' % dm_exclude
      ]
  }

  # Highlight exclusions.
  he = (
      'not(contains(@class, "cnnHiliteHeader"))'
      ' and not(descendant::*[starts-with(text(), "Next Article in")])')
  highlight_selectors = {
      'cnn': [
          '//*[contains(@class, "el__storyhighlights__list")]//li[%s]' % he,
          '//*[contains(@class, "cnnStryHghLght")]//li[%s]' % he,
          '//*[@id="cnnHeaderRightCol"]//li[%s]' % he
      ],
      'dailymail': [
          '//h1/following-sibling::ul//li'
      ]
  }

  def ExtractText(selector):
    """Extracts a list of paragraphs given a XPath selector.
    Args:
      selector: A XPath selector to find the paragraphs.
    Returns:
      A list of raw text paragraphs with leading and trailing whitespace.
    """

    xpaths = map(tree.xpath, selector)
    elements = list(chain.from_iterable(xpaths))
    #paragraphs = [str(e.text_content().encode('utf-8')) for e in elements]
    paragraphs = [e.text_content() for e in elements]


    # Remove editorial notes, etc.
    if corpus == 'cnn' and len(paragraphs) >= 2 and '(CNN)' in paragraphs[1]:
      paragraphs.pop(0)

    paragraphs = map(str.strip, paragraphs)
    paragraphs = [s for s in paragraphs if s and not str.isspace(s)]

    return paragraphs

  for selector in delete_selectors[corpus]:
    for bad in tree.xpath(selector):
      bad.getparent().remove(bad)

  paragraphs = ExtractText(paragraph_selectors[corpus])
  highlights = ExtractText(highlight_selectors[corpus])

  content = '\n\n'.join(paragraphs)

  return Story(story.url, content, highlights)







def StoreMapper(path, file_id, corpus):
  """Reads an URL from disk and returns the parsed news story.
  Args:
    t: a tuple (url, corpus).
  Returns:
    A Story containing the parsed news story.
  """
  char_type = chardet.detect(open(path+file_id, 'rb').read())['encoding']
  story_html = open(path+file_id, 'rb').read()

  if not story_html:
    return None

  raw_story = RawStory(file_id, story_html)

  return ParseHtml(raw_story, corpus), char_type


def load_and_parse_cnn(path, file_name):
    article = dict()
    story, char_type = StoreMapper(path, file_name, 'cnn')
    article['summary_items'] = [s.strip().replace('NEW:', "").strip() for s in story.highlights if len(s) > 10]
    article['text_content'] = " ".join([s.strip() for s in story.content.split("\n") if len(s) > 10])
    if '(CNN)' in article['text_content']:
        article['text_content'] = "".join(article['text_content'].split('(CNN)')[1:])
    article['headline'] = ''
    article['timestamp_pub'] = '0'
    return article



import json, glob
from multiprocessing import Process


def decode_and_write_cnn_to_json(file_names, path, out_path, p_id):
    count = 0
    data = dict()
    for file_name in file_names:
        count += 1
        if count % 1000 == 0:
            print(p_id, count)
            with open(out_path+"CNN_"+str(int(count/1000)) + "000_"+str(p_id)+".txt", "w") as handle:
                handle.write(json.dumps(data))
                pass
            data = dict()

        try:
            data[file_name] = load_and_parse_cnn(path, file_name)
        except:
            print("Error with decoding file")

    if len(data) != 0:
        with open(out_path + "CNN_last" + str(p_id) + ".txt", "w") as handle:
            handle.write(json.dumps(data))



if __name__ == '__main__':
    path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/CNN/downloads/'
    out_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/CNN/extracted/'
    file_names = [f[len(path):] for f in list(glob.iglob(path+"*.html"))]

    nb_processes = 8
    tasks = []
    for i in range(nb_processes): tasks.append([])
    processes = []
    for i in range(len(file_names)): tasks[i % nb_processes].append(file_names[i])
    for p in range(nb_processes):
        proc = Process(target=decode_and_write_cnn_to_json, args=(tasks[p], path, out_path, p))
        proc.start()
        processes.append(proc)
    for p in processes: p.join()











