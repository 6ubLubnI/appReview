from review import appReview
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import string
from itertools import compress
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Chrome
from lxml import etree
import re
from wordcloud import WordCloud
from PIL import Image
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from textblob import TextBlob
import en_core_web_md

'''
#################################################################################################
to scrap the web data
#################################################################################################
'''
app_review = appReview(Path(r'C:\Webdriver\bin\chromedriver.exe'), True)
play_store_review = app_review.get_play_store('com.vitality.aia.hk', 10)
app_store_review = app_review.get_app_store('917286353', 10)


'''
#################################################################################################
distribution plot
#################################################################################################
'''
app_store_review['os'] = 'ios'
play_store_review['os'] = 'android'
all_review = pd.concat([app_store_review.drop('title', axis=1), play_store_review], axis=0)
app_store_review['yrmh'] = [d.year*100+d.month for d in app_store_review.date]
play_store_review['yrmh'] = [d.year*100+d.month for d in play_store_review.date]
all_review['yrmh'] = [d.year*100+d.month for d in all_review.date]

app_yrmh = app_store_review.groupby('yrmh')['star'].mean()
play_yrmh = play_store_review.groupby('yrmh')['star'].mean()
all_yrmh = all_review.groupby('yrmh')['star'].mean()

play_yrmh[201511] = (play_yrmh[201510] + play_yrmh[201512])/2
play_yrmh[201602] = (play_yrmh[201601] + play_yrmh[201603])/2
play_yrmh[201604] = (play_yrmh[201603] + play_yrmh[201605])/2
play_yrmh[201910] = (play_yrmh[201909] + play_yrmh[201911])/2
play_yrmh.sort_index(inplace=True)

app_yrmh[201511] = (app_yrmh[201510] + app_yrmh[201601])/2
app_yrmh[201512] = (app_yrmh[201510] + app_yrmh[201601])/2
app_yrmh[201603] = (app_yrmh[201602] + app_yrmh[201604])/2
app_yrmh[201605] = (app_yrmh[201604] + app_yrmh[201607])/2
app_yrmh[201606] = (app_yrmh[201604] + app_yrmh[201607])/2
app_yrmh[201807] = (app_yrmh[201806] + app_yrmh[201808])/2
app_yrmh.sort_index(inplace=True)

all_yrmh[201511] = (all_yrmh[201510] + all_yrmh[201512])/2
all_yrmh.sort_index(inplace=True)


plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Star', fontsize=12)
plt.title('App Review', fontsize=24)
plt.xticks(list(range(len(play_yrmh))), play_yrmh.index.values, rotation=45)
play, = plt.plot(play_yrmh.values, color='tab:green', marker='x', linewidth=2, label='Play Store Review')
app, = plt.plot(app_yrmh.values, color='tab:blue', marker='x', linewidth=2, label='App Store Review')
plt.legend(handles=[play, app])

plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Star', fontsize=12)
plt.title('App Review', fontsize=24)
plt.xticks(list(range(len(play_yrmh))), play_yrmh.index.values, rotation=45)
play_smoothed, = plt.plot(play_yrmh.rolling(window=6).mean().values, color='mediumaquamarine', marker='x', linewidth=3, linestyle='dashed', alpha=1, label='Play Store Review Smoothed')
app_smoothed, = plt.plot(app_yrmh.rolling(window=6).mean().values, color='cornflowerblue', marker='x', linewidth=3, linestyle='dashed', alpha=1, label='App Store Review Smoothed')
plt.legend(handles=[play_smoothed, app_smoothed])

plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Star', fontsize=12)
plt.title('App Review', fontsize=24)
plt.xticks(list(range(len(all_yrmh))), all_yrmh.index.values, rotation=45)
all_p, = plt.plot(all_yrmh.values, color='cadetblue', marker='x', linewidth=2, label='All App Review')
all_smoothed, = plt.plot(all_yrmh.rolling(window=6).mean().values, color='salmon', marker='x', linewidth=3, linestyle='dashed', alpha=0.5, label='All App Review Smoothed')
plt.legend(handles=[all_p, all_smoothed])


'''
#################################################################################################
chinese translation to english
#################################################################################################
'''
def translate2en(review_data, webdriver_path, headless, to):
    unstandard_len = []
    for review in review_data.review:
        unstandard_len.append(len([r for r in review if r not in '…‘’“”！,。？'+string.printable]))

    tobe_translated = list(compress(review_data.review, [le>=1 for le in unstandard_len]))
    tobe_translated = [re.sub('[。！？&]', ',', s) for s in tobe_translated]
    tobe_translated_len = [len(s) for s in tobe_translated]
    tobe_translated_len_group = [int(g) for g in np.ceil(np.cumsum(tobe_translated_len)/4500)]
    index_tobe_translated = [i for i,v in zip(review_data.index, [le>=1 for le in unstandard_len]) if v]

    output_translated = []
    for i in range(1, np.max(tobe_translated_len_group)+1):
        group_review = "%0A".join(list(compress(tobe_translated, [g==i for g in tobe_translated_len_group])))

        chrome_options = Options()
        chrome_options.headless = headless

        with Chrome(str(webdriver_path), options=chrome_options) as driver:
            driver.get(f'https://translate.google.com/#view=home&op=translate&sl=zh-CN&tl={to}&text={group_review}')
            html = etree.HTML(driver.page_source)
            translated = [str(w) for w in html.xpath('//span[@class="tlid-translation translation"]/text()')]
            if len(translated)==0:
                translated = [str(w) for w in html.xpath('//span[@class="tlid-translation translation"]/span/text()')]
        output_translated.extend(translated)
        assert len(translated)==sum([g==i for g in tobe_translated_len_group])

    review_data.loc[index_tobe_translated, 'review'] = output_translated
    return review_data
play_store_translated = translate2en(play_store_review, Path(r'C:\Webdriver\bin\chromedriver.exe'), True, 'en')
app_store_translated = translate2en(app_store_review, Path(r'C:\Webdriver\bin\chromedriver.exe'), True, 'en')


'''
#################################################################################################
text processing: tokenization, removing stop words, lemmetization
#################################################################################################
'''
nlp = en_core_web_md.load()
def tokenize(review):
    token = []
    pos = []
    for r in review:
        doc = nlp(r)
        token.extend([d.lemma_ for d in doc if (d.lemma_ not in STOP_WORDS) and (d.pos_!='PUNCT') and (d.lemma_ not in '…‘’“”！,。？！+-')])
        pos.extend([d.pos_ for d in doc if (d.lemma_ not in STOP_WORDS) and (d.pos_!='PUNCT') and (d.lemma_ not in '…‘’“”！,。？！+-')])
    return token, pos
play_token, play_pos = tokenize(play_store_translated.review)
app_token, app_pos = tokenize(app_store_translated.review)



'''
#################################################################################################
wordcloud plot
#################################################################################################
'''
aia_mask = np.array(Image.open("AIA.png"))
def plot_wordcloud(token, mask, pic_name, relative_scaling):
    wc_dict = dict(zip(*np.unique(token, return_counts=True)))
    wc_dict.pop('-PRON-', None)
    wc = WordCloud(max_font_size=40, min_font_size=4, relative_scaling=relative_scaling, max_words=400, width=1600, height=900, scale=16, mask=mask, background_color='white').fit_words(wc_dict)
    plt.figure(figsize=(20,12))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.savefig(f'{pic_name}.png', facecolor='k', bbox_inches='tight')
    return None
plot_wordcloud(play_token, aia_mask, 'play_store_wordcloud', 0.5)
plot_wordcloud(app_token, aia_mask, 'app_store_wordcloud', 0.5)
plot_wordcloud(play_token + app_token, aia_mask, 'all_store_wordcloud', 0.5)


'''
#################################################################################################
POS pie chart plot
#################################################################################################
'''
def plot_pos(pos, pic_name):
    pos_dict = dict(zip(*np.unique(pos, return_counts=True)))
    pos_sorted = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(10,10))
    plt.pie([y for x,y in pos_sorted], labels=[x for x,y in pos_sorted], autopct='%1.1f%%', shadow=False, startangle=140)
    plt.axis('equal')
    # plt.savefig(f'{pic_name}.png')
    return None
plot_pos(play_pos, 'play_pos')
plot_pos(app_pos, 'app_pos')
plot_pos(play_pos + app_pos, 'all_pos')


'''
#################################################################################################
dependency parsing plot
#################################################################################################
'''
doc = nlp(play_store_translated.review[1])
displacy.render(doc, style='dep', jupyter=True)


'''
#################################################################################################
POS analysis
#################################################################################################
'''
play_adj = list(compress(play_token, [pos=='ADJ' for pos in play_pos]))
app_adj = list(compress(app_token, [pos=='ADJ' for pos in app_pos]))
def adj_polarity(adj):
    adj_dict = dict(zip(*np.unique(adj, return_counts=True)))
    for k in adj_dict:
        adj_dict[k] = [adj_dict[k], TextBlob(k).sentiment.polarity]
    count_sorted = sorted(adj_dict.items(), key=lambda x: x[1][0], reverse=True)
    return count_sorted
play_adj_polarity = adj_polarity(play_adj)
app_adj_polarity = adj_polarity(app_adj)
all_adj_polarity = adj_polarity(play_adj + app_adj)


play_verb = list(compress(play_token, [pos=='VERB' for pos in play_pos]))
app_verb = list(compress(app_token, [pos=='VERB' for pos in app_pos]))
def adj_polarity(adj):
    adj_dict = dict(zip(*np.unique(adj, return_counts=True)))
    for k in adj_dict:
        adj_dict[k] = [adj_dict[k], TextBlob(k).sentiment.polarity]
    count_sorted = sorted(adj_dict.items(), key=lambda x: x[1][0], reverse=True)
    return count_sorted
all_verb_polarity = adj_polarity(play_verb + app_verb)


'''
#################################################################################################
review samples
#################################################################################################
'''
play_store_translated['polarity'] = [TextBlob(r).sentiment.polarity for r in play_store_translated.review]
for i in play_store_translated.sort_values(by='polarity', ascending=False).iloc[:5, :].review:
    print(i)


app_store_translated['polarity'] = [TextBlob(r).sentiment.polarity for r in app_store_translated.review]
for i in app_store_translated.sort_values(by='polarity', ascending=False).iloc[:5, :].review:
    print(i)
