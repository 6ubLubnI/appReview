from datetime import datetime
from review import appReview
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import string
import time
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
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine


Path(Path().cwd()/'data').mkdir(parents=True, exist_ok=True)
CHROMEDRIVER_PATH = Path(r'C:\Webdriver\bin\chromedriver.exe')

'''
#################################################################################################
to scrap the web data
#################################################################################################
'''
# translation of chinese to english
def translate2en(review_data, webdriver_path, headless, to):
    unstandard_len = []
    for review in review_data.review:
        unstandard_len.append(len([r for r in review if r not in '…‘’“”！,。？'+string.printable]))

    tobe_translated = list(compress(review_data.review, [le>=1 for le in unstandard_len]))
    tobe_translated = [re.sub('[。，！？&?/]', '.', s) for s in tobe_translated]
    tobe_translated = [re.sub('\n', '', s) for s in tobe_translated]
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

        try:
            assert len(translated)==sum([g==i for g in tobe_translated_len_group])
        except:
            with Chrome(str(webdriver_path), options=chrome_options) as driver:
                for i in list(compress(tobe_translated, [g==i for g in tobe_translated_len_group])):
                    driver.get(f'https://translate.google.com/#view=home&op=translate&sl=auto&tl={to}&text={i}')
                    time.sleep(1)
                    html = etree.HTML(driver.page_source)
                    translated = ' '.join([str(w) for w in html.xpath('//span[@class="tlid-translation translation"]/text()')])
                    if len(re.sub(' ', '', translated))==0:
                        translated = ' '.join([str(w) for w in html.xpath('//span[@class="tlid-translation translation"]/span/text()')])
                    output_translated.append(translated)
    review_data.loc[index_tobe_translated, 'review'] = output_translated
    return review_data


app_review = appReview(webdriver_path=CHROMEDRIVER_PATH, headless=True)

# AIA Vitality
play_store_reviews = pd.concat([get_loc_review('en'), get_loc_review('zh_HK')], axis=0).reset_index()
play_store_review = app_review.get_play_store('com.vitality.aia.hk', 10)
app_store_review = app_review.get_app_store('917286353', 10)

play_store_translated = translate2en(play_store_review, CHROMEDRIVER_PATH, True, 'en')
app_store_translated = translate2en(app_store_review, CHROMEDRIVER_PATH, True, 'en')

app_store_review['os'] = 'ios'
play_store_review['os'] = 'android'
all_review_translated = pd.concat([play_store_translated, app_store_translated.drop('title', axis=1)], axis=0).sort_values('date')
all_review_translated.to_csv(Path().cwd()/'data'/'vitality.txt', index=False)


# AIA Connect
play_store_review = app_review.get_play_store('com.aiahk.idirect', 10)
app_store_review = app_review.get_app_store('1292514758', 10)

play_store_translated = translate2en(play_store_review, CHROMEDRIVER_PATH, True, 'en')
app_store_translated = translate2en(app_store_review, CHROMEDRIVER_PATH, True, 'en')

app_store_review['os'] = 'ios'
play_store_review['os'] = 'android'
all_review_translated = pd.concat([play_store_translated, app_store_translated.drop('title', axis=1)], axis=0).sort_values('date')
all_review_translated.to_csv(Path().cwd()/'data'/'aia_connect.txt', index=False)


# AIA CN App
app_store_review = app_review.get_app_store('1207516706', 10, country='cn')
app_store_translated = translate2en(app_store_review, Path(r'C:\Webdriver\bin\chromedriver.exe'), True, 'en')
app_store_translated.to_csv(Path().cwd()/'data'/'cn_app.txt', index=False)


# Manulife Move
play_store_review = app_review.get_play_store('com.manulife.move', 10)
app_store_review = app_review.get_app_store('1031487052', 10)

play_store_translated = translate2en(play_store_review, Path(r'C:\Webdriver\bin\chromedriver.exe'), True, 'en')
app_store_translated = translate2en(app_store_review, Path(r'C:\Webdriver\bin\chromedriver.exe'), True, 'en')

app_store_review['os'] = 'ios'
play_store_review['os'] = 'android'
all_review_translated = pd.concat([play_store_translated, app_store_translated.drop('title', axis=1)], axis=0).sort_values('date')
all_review_translated.to_csv(Path().cwd()/'data'/'move.txt', index=False)


# read in the reviews
vitality = pd.read_csv(Path.cwd()/'data'/'vitality.txt')
aia_connect = pd.read_csv(Path.cwd()/'data'/'aia_connect.txt')
move = pd.read_csv(Path.cwd()/'data'/'move.txt')
cn_app = pd.read_csv(Path.cwd()/'data'/'cn_app.txt')


# select only the reviews in or after 2019
vitality = vitality[np.array([int(d[:4])>=2019 for d in vitality.date])]
aia_connect = aia_connect[np.array([int(d[:4])>=2019 for d in aia_connect.date])]
move = move[np.array([int(d[:4])>=2019 for d in move.date])]
cn_app = cn_app[np.array([int(d[:4])>=2019 for d in cn_app.date])]

'''
###############################################################################################################
# review count by month
###############################################################################################################
'''
# vitality['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in vitality.date]
# vitality['yrmh'] = [d.year*100+d.month for d in vitality.date]
# aia_connect['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in aia_connect.date]
# aia_connect['yrmh'] = [d.year*100+d.month for d in aia_connect.date]
# move['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in move.date]
# move['yrmh'] = [d.year*100+d.month for d in move.date]
# cn_app['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in cn_app.date]
# cn_app['yrmh'] = [d.year*100+d.month for d in cn_app.date]
vitality['yrmh'] = [d[:7] for d in vitality.date]
aia_connect['yrmh'] = [d[:7] for d in aia_connect.date]
move['yrmh'] = [d[:7] for d in move.date]
cn_app['yrmh'] = [d[:7] for d in cn_app.date]

vitality_yrmh_count = vitality.groupby('yrmh', as_index=False)['review'].count().rename(columns={'review': 'vitality'})
aia_connect_yrmh_count = aia_connect.groupby('yrmh', as_index=False)['review'].count().rename(columns={'review': 'aia_connect'})
move_yrmh_count = move.groupby('yrmh', as_index=False)['review'].count().rename(columns={'review': 'move'})
cn_app_yrmh_count = cn_app.groupby('yrmh', as_index=False)['review'].count().rename(columns={'review': 'cn_app'})
vitality_yrmh_count.merge(aia_connect_yrmh_count, on='yrmh', how='outer')\
    .merge(move_yrmh_count, on='yrmh', how='outer')\
    .merge(cn_app_yrmh_count, on='yrmh', how='outer').sort_values('yrmh')

'''
###############################################################################################################
# rating trend by month
###############################################################################################################
'''
vitality_yrmh_star = vitality.groupby('yrmh', as_index=False)['star'].mean().rename(columns={'star': 'vitality'})
aia_connect_yrmh_star = aia_connect.groupby('yrmh', as_index=False)['star'].mean().rename(columns={'star': 'aia_connect'})
cn_app_yrmh_star = cn_app.groupby('yrmh', as_index=False)['star'].mean().rename(columns={'star': 'cn_app'})
move_yrmh_star = move.groupby('yrmh', as_index=False)['star'].mean().rename(columns={'star': 'move'})

all_star = vitality_yrmh_star.merge(aia_connect_yrmh_star, on='yrmh', how='outer')\
    .merge(move_yrmh_star, on='yrmh', how='outer')\
    .merge(cn_app_yrmh_star, on='yrmh', how='outer')
for col in ['vitality', 'aia_connect', 'cn_app', 'move']:
    for n, row in enumerate(all_star[col]):
        if n!=0 and np.isnan(row) and n!=(all_star.shape[0]-1):
            all_star[col][n] = (all_star[col][n+1]+all_star[col][n-1])/2
all_star.sort_values('yrmh', inplace=True)

plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Star', fontsize=12)
plt.title('App Review', fontsize=24)
plt.xticks(list(range(len(all_star))), all_star.yrmh, rotation=45)
v, = plt.plot(all_star.vitality.values, color='tab:orange', marker='x', linewidth=2, label='AIA Vitality Review Star')
a, = plt.plot(all_star.aia_connect.values, color='tab:red', marker='x', linewidth=2, label='AIA Connect Review Star')
c, = plt.plot(all_star.cn_app.values, color='tab:blue', marker='x', linewidth=2, label='AIA CN App Review Star')
m, = plt.plot(all_star.move.values, color='tab:green', marker='x', linewidth=2, label='Manulife Move Review Star')
plt.legend(handles=[v, a, c, m])
plt.savefig(Path.cwd()/'graphs'/'all_star_raw.png')


plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Star', fontsize=12)
plt.title('App Review', fontsize=24)
plt.xticks(list(range(len(all_star))), all_star.yrmh, rotation=45)
v, = plt.plot(all_star.vitality.rolling(window=6).mean().values, color='tab:orange', marker='x', linewidth=3, linestyle='dashed', alpha=0.5, label='AIA Vitality Review Star')
a, = plt.plot(all_star.aia_connect.rolling(window=6).mean().values, color='tab:red', marker='x', linewidth=3, linestyle='dashed', alpha=0.5, label='AIA Connect Review Star')
c, = plt.plot(all_star.cn_app.rolling(window=6).mean().values, color='tab:blue', marker='x', linewidth=3, linestyle='dashed', alpha=0.5, label='AIA CN App Review Star')
m, = plt.plot(all_star.move.rolling(window=6).mean().values, color='tab:green', marker='x', linewidth=3, linestyle='dashed', alpha=0.5, label='Manulife Move Review Star')
plt.legend(handles=[v, a, c, m])
plt.savefig(Path.cwd()/'graphs'/'all_star_smoothed.png')


'''
###############################################################################################################
# wordcloud
###############################################################################################################
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
vitality_token, vitality_pos = tokenize(vitality.review)
aia_connect_token, aia_connect_pos = tokenize(aia_connect.review)
cn_app_token, cn_app_pos = tokenize(cn_app.review)
move_token, move_pos = tokenize(move.review)

mask = np.array(Image.open("./misc/bubble.jpg"))
def plot_wordcloud(token, mask, pic_name, relative_scaling):
    wc_dict = dict(zip(*np.unique(token, return_counts=True)))
    wc_dict.pop('-PRON-', None)
    # wc = WordCloud(max_font_size=40, min_font_size=4, relative_scaling=relative_scaling, max_words=400, width=1600, height=900, scale=16, mask=mask, background_color='white').fit_words(wc_dict)
    wc = WordCloud(max_font_size=60, min_font_size=5, relative_scaling=relative_scaling, max_words=400, width=800, height=800, scale=4, mask=mask, background_color='white').fit_words(wc_dict)
    plt.figure(figsize=(10,10))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f'./graphs/{pic_name}.png', facecolor='k', bbox_inches='tight')
    return None
plot_wordcloud(vitality_token, mask, 'AIA Vitality Wordcloud', 0.2)
plot_wordcloud(aia_connect_token, mask, 'AIA Connect Wordcloud', 0.2)
plot_wordcloud(cn_app_token, mask, 'AIA CN App Wordcloud', 0.2)
plot_wordcloud(move_token, mask, 'Manulife Move Wordcloud', 0.2)





'''
###############################################################################################################
# POS distribution
###############################################################################################################
'''
def plot_pos(pos, pic_name):
    pos_dict = dict(zip(*np.unique(pos, return_counts=True)))
    pos_sorted = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)

    def my_autopct(pct):
        return ('%.1f%%' % pct) if pct > 5 else None

    plt.figure(figsize=(10,10))
    plt.pie([y for x,y in pos_sorted], labels=[x if y/np.sum([y for x,y in pos_sorted])>0.05 else None for x,y in pos_sorted], autopct=my_autopct, shadow=False, startangle=140, textprops={'fontsize': 18})
    plt.axis('equal')
    plt.savefig(f'./graphs/{pic_name}.png')
    return None
plot_pos(vitality_pos, 'vitality_pos')
plot_pos(aia_connect_pos, 'aia_connect_pos')
plot_pos(cn_app_pos, 'cn_app_pos')
plot_pos(move_pos, 'move_pos')


'''
#################################################################################################
POS analysis
#################################################################################################
'''
def pos_polarity(adj):
    pos_dict = dict(zip(*np.unique(adj, return_counts=True)))
    for k in pos_dict:
        pos_dict[k] = [pos_dict[k], TextBlob(k).sentiment.polarity]
    count_sorted = sorted(pos_dict.items(), key=lambda x: x[1][0], reverse=True)
    return count_sorted

vitality_adj = list(compress(vitality_token, [pos=='ADJ' for pos in vitality_pos]))
vitality_verb = list(compress(vitality_token, [pos=='VERB' for pos in vitality_pos]))
vitality_adj_polarity = pos_polarity(vitality_adj)
vitality_verb_polarity = pos_polarity(vitality_verb)

aia_connect_adj = list(compress(aia_connect_token, [pos=='ADJ' for pos in aia_connect_pos]))
aia_connect_verb = list(compress(aia_connect_token, [pos=='VERB' for pos in aia_connect_pos]))
aia_connect_adj_polarity = pos_polarity(aia_connect_adj)
aia_connect_verb_polarity = pos_polarity(aia_connect_verb)

cn_app_adj = list(compress(cn_app_token, [pos=='ADJ' for pos in cn_app_pos]))
cn_app_verb = list(compress(cn_app_token, [pos=='VERB' for pos in cn_app_pos]))
cn_app_adj_polarity = pos_polarity(cn_app_adj)
cn_app_verb_polarity = pos_polarity(cn_app_verb)

move_adj = list(compress(move_token, [pos=='ADJ' for pos in move_pos]))
move_verb = list(compress(move_token, [pos=='VERB' for pos in move_pos]))
move_adj_polarity = pos_polarity(move_adj)
move_verb_polarity = pos_polarity(move_verb)

def plot_adj(ax, data, title):
    ax.bar(range(10), [s[0] for w,s in data][:10]/np.sum([s[0] for w,s in data]), color='tab:blue', alpha=0.5)
    ax_2 = ax.twinx()
    ax_2.plot([s[1] for w,s in data][:10], marker='.', color='tab:green', alpha=0.8)
    ax_2.axhline(y=0, linestyle='--', color='tab:grey', alpha=0.2)
    ax.set_xticks(range(10))
    ax.set_xticklabels([w for w,s in data][:10])
    ax.set_ylabel('Proportion')
    ax_2.set_ylabel('Polarity')
    ax.set_title(f'{title} Top 10 Ajective')
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in y_ticks])
    return ax, ax_2
def plot_adj_all(data1, data2, data3, data4):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=[15, 20])
    ax1, ax1_2 = plot_adj(ax1, data1, 'AIA Vitality')
    ax2, ax2_2 = plot_adj(ax2, data2, 'AIA Connect')
    ax3, ax3_2 = plot_adj(ax3, data3, 'AIA CN App')
    ax4, ax4_2 = plot_adj(ax4, data4, 'Manulife Move')
    return fig
fig_adj = plot_adj_all(vitality_adj_polarity, aia_connect_adj_polarity, cn_app_adj_polarity, move_adj_polarity)
fig_adj.savefig(Path.cwd()/'graphs'/'adj_polarity.png')

def plot_verb(ax, data, title):
    ax.bar(range(10), [s[0] for w,s in data][:10]/np.sum([s[0] for w,s in data]), color='tab:blue', alpha=0.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels([w for w,s in data][:10])
    ax.set_ylabel('Proportion')
    ax.set_title(f'{title} Top 10 Verbs')
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in y_ticks])
    return ax
def plot_verb_all(data1, data2, data3, data4):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=[15, 20])
    ax1 = plot_verb(ax1, data1, 'AIA Vitality')
    ax2 = plot_verb(ax2, data2, 'AIA Connect')
    ax3 = plot_verb(ax3, data3, 'AIA CN App')
    ax4 = plot_verb(ax4, data4, 'Manulife Move')
    return fig
fig_verb = plot_verb_all(vitality_verb_polarity, aia_connect_verb_polarity, cn_app_verb_polarity, move_verb_polarity)
fig_verb.savefig(Path.cwd()/'graphs'/'verb.png')

'''
###############################################################################################################
# text summarization using BERT
###############################################################################################################
'''
def convert_to_inputs(vocab_file, reviews):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    review_tokens = [tokenizer.tokenize(review) for review in reviews]
    review_tokens2 = [['[CLS]'] + tokens + ['[SEP]'] for tokens in review_tokens]
    token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in review_tokens2]

    max_len = max([len(id) for id in token_ids])
    input_ids = [ids+[0]*(max_len-len(ids)) for ids in token_ids]
    input_mask = [[0]*len(ids)+[1]*(max_len-len(ids)) for ids in input_ids]
    segment_ids = [[0]*max_len for mask in input_mask]
    return max_len, [input_ids, input_mask, segment_ids]

def bert_embed(max_len, inputs):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
    return model.predict(inputs)

vitality_embed, vitality_token_embed = bert_embed(*convert_to_inputs(vocab_file='./vocab.txt', reviews=vitality.review))
aia_connect_embed, aia_connect_token_embed = bert_embed(*convert_to_inputs(vocab_file='./vocab.txt', reviews=aia_connect.review))
move_embed, move_token_embed = bert_embed(*convert_to_inputs(vocab_file='./vocab.txt', reviews=move.review))
cn_app_embed, cn_token_embed = bert_embed(*convert_to_inputs(vocab_file='./vocab.txt', reviews=cn_app.review))


def find_cluster(data, embeddings, cluster_num):
    distortions = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=1000, n_jobs=-1, random_state=32746)
        km.fit(embeddings)
        distortions.append(km.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')

    km = KMeans(n_clusters=cluster_num, max_iter=1000, n_jobs=-1)
    data['cluster'] = km.fit_predict(embeddings)
    return data, km
vitality, vitality_km = find_cluster(vitality, vitality_embed, 5)
aia_connect, aia_connect_km = find_cluster(aia_connect, aia_connect_embed, 5)
move, move_km = find_cluster(move, move_embed, 5)
cn_app, cn_app_km = find_cluster(cn_app, cn_app_embed, 5)

nlp = en_core_web_md.load()
def tokenize(review):
    token = []
    pos = []
    for r in review:
        doc = nlp(r)
        token.append([d.lemma_ for d in doc if (d.lemma_ not in STOP_WORDS) and (d.pos_!='PUNCT') and (d.lemma_ not in '…‘’“”！,。？！+-')])
        pos.append([d.pos_ for d in doc if (d.lemma_ not in STOP_WORDS) and (d.pos_!='PUNCT') and (d.lemma_ not in '…‘’“”！,。？！+-')])
    return token, pos


vitality_clusters = [list(zip(token, pos)) for token, pos in vitality.groupby('cluster')['review'].apply(tokenize)]
aia_connect_clusters = [list(zip(token, pos)) for token, pos in aia_connect.groupby('cluster')['review'].apply(tokenize)]
cn_app_clusters = [list(zip(token, pos)) for token, pos in cn_app.groupby('cluster')['review'].apply(tokenize)]
move_clusters = [list(zip(token, pos)) for token, pos in move.groupby('cluster')['review'].apply(tokenize)]

# top five representative reviews
def center_review(data, embed, km, clusters, cluster):
    dist2center = np.apply_along_axis(cosine, 1, embed[data.cluster==cluster], v=km.cluster_centers_[cluster])
    cluster_reviews = data.review[data.cluster==cluster]
    cluster_reviews_tokens = [t for t,p in clusters[cluster]]

    sorted_cluster_reviews = sorted(zip(dist2center, cluster_reviews, cluster_reviews_tokens), key=lambda x: x[0])
    print([r for d, r, t in sorted_cluster_reviews][0])
    # for i in [r for d, r, t in sorted_cluster_reviews][:3]:
    #     print(i)
    # for i in sorted(zip(*np.unique([l2 for l1 in [t for d, r, t in sorted_cluster_reviews][:10] for l2 in l1], return_counts=True)), key=lambda x: x[1], reverse=True)[:10]:
    #     print(i)
for i in range(5):
    center_review(vitality, vitality_embed, vitality_km, vitality_clusters, i)
for i in range(5):
    center_review(aia_connect, aia_connect_embed, aia_connect_km, aia_connect_clusters, i)
for i in range(5):
    center_review(cn_app, cn_app_embed, cn_app_km, cn_app_clusters, i)
for i in range(5):
    center_review(move, move_embed, move_km, move_clusters, i)




# for embedding illustration
new = []
for v in move.review:
    if '\n' in v:
        new.append(v.replace('\n', ''))
    else:
        new.append(v)
pd.concat([pd.DataFrame(new, columns=['review']), move['star']], axis=1).to_csv('move_meta.tsv', sep='\t', index=False)
pd.DataFrame(move_embed).to_csv('move_embed.tsv', sep='\t', index=False, header=False)
