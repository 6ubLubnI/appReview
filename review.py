from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from lxml import etree
import pandas as pd
from itertools import compress
from datetime import datetime as dt
import time
import re
from win32api import GetSystemMetrics


class appReview(object):
    def __init__(self, webdriver_path, headless=True):
        self.webdriver_path = webdriver_path
        self.chrome_options = Options()
        self.chrome_options.headless = headless

    def get_play_store(self, app_id, wait):
        '''extract the reviews from google store
        Arguments:
        app_id(string): the google play store app id
        wait(float): seconds to wait for loading the extra reviews before termination
        Return:
        reviews(pd.DataFrame): the extracted review data
        '''
        def get_loc_review(loc):
            with Chrome(str(self.webdriver_path), options=self.chrome_options) as driver:
                driver.get(f'https://play.google.com/store/apps/details?id={app_id}&hl={loc}&showAllReviews=true')
                driver.set_window_size(GetSystemMetrics(0), GetSystemMetrics(1))

                def show_more(driver):
                    '''This is to locate "show more" button element in html'''
                    driver.find_element_by_tag_name('body').send_keys(Keys.END)
                    driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
                    driver.find_element_by_tag_name('body').send_keys(Keys.END)
                    return driver.find_element_by_xpath('//div[@class="PFAhAf"]/div')

                # loop over the action to locate "show more" button and click for more
                while True:
                    try:
                        show_more_button = WebDriverWait(driver, timeout=wait, ignored_exceptions=[NoSuchElementException]).until(show_more)
                        show_more_button.click()
                    except TimeoutException:
                        show_more_button = None
                    if show_more_button is None:
                        break

                # parse the extracted content
                html = etree.HTML(driver.page_source)
                id = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]/@jsdata')]
                name = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]//div[@class="xKpxId zc7KVe"]//span[@class="X43Kjb"]/text()')]
                star = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]//span[@class="nt2C1d"]/div[@class="pf5lIe"]/div/@aria-label')]
                date = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]//div[@class="xKpxId zc7KVe"]//span[@class="p2TkOb"]/text()')]
                short_review = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]//div[@class="UD7Dzf"]/span[@jsname="bN97Pc"]/text()')]
                long_review = [str(w) for w in html.xpath('//div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]//div[@class="UD7Dzf"]/span[@jsname="fbQN7e"]/text()')]
                long_review_id = [str(w) for w in html.xpath('//div[@class="UD7Dzf"]/span[@jsname="fbQN7e"]/text()/ancestor::div[@jscontroller="H6eOGe"][@jsmodel="y8Aajc"]/@jsdata')]

                # to replace the short review with long review content
                for n, i in enumerate(id):
                    if i in long_review_id:
                        short_review[n] = list(compress(long_review, [j==i for j in long_review_id]))[0]

                play_store_reviews = pd.DataFrame({'name': name, 'date': date, 'review': short_review, 'star': star})
                if loc=='en':
                    play_store_reviews['date'] = [dt.strptime(d, '%B %d, %Y').date() for d in play_store_reviews.date]
                elif loc=='zh_HK':
                    play_store_reviews['date'] = [dt.strptime(d, '%Y年%m月%d日').date() for d in play_store_reviews.date]
                play_store_reviews['star'] = [int(re.findall('\d', rate)[0]) for rate in play_store_reviews.star]
                play_store_reviews.sort_values(by='date', ascending=False, inplace=True)
                play_store_reviews.reset_index(drop=True, inplace=True)
                return play_store_reviews
        play_store_reviews = pd.concat([get_loc_review('en'), get_loc_review('zh_HK')], axis=0).reset_index(drop=True)
        return play_store_reviews

    def get_app_store(self, app_id, wait, country='hk'):
        '''extract the reviews from app store
        Arguments:
        app_id(string): the app play store app id
        wait(float): seconds to wait for loading the extra reviews before termination
        Return:
        reviews(pd.DataFrame): the extracted review data
        '''

        with Chrome(str(self.webdriver_path), options=self.chrome_options) as driver:
            driver.get(f'https://apps.apple.com/{country}/app/id{app_id}?l=en#see-all/reviews')
            driver.maximize_window()

            # this is to scroll through the website to extract more
            while True:
                initial_website_height = driver.execute_script('return document.body.scrollHeight')
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)

                initial_time = time.time()
                while True:
                    current_website_height = driver.execute_script('return document.body.scrollHeight')
                    if current_website_height!=initial_website_height:
                        break
                    current_time = time.time()
                    if current_time-initial_time>wait:
                        break

                if current_website_height==initial_website_height:
                    break

            # text extraction by xpath
            html = etree.HTML(driver.page_source)
            title = [str(w) for w in html.xpath('//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//h3[@class="we-truncate we-truncate--single-line ember-view we-customer-review__title"]/text()')]
            date = [str(w) for w in html.xpath('//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//time/text()')]
            name = [str(w) for w in html.xpath('//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//span[@class="we-truncate we-truncate--single-line ember-view we-customer-review__user"]/text()')]
            star = [str(w) for w in html.xpath('//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//figure/@aria-label')]
            review_id = [str(w) for w in html.xpath('//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//blockquote[1]/@id')]
            reviews = []
            for id in review_id:
                review = [str(w) for w in html.xpath(f'//div[@class="ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"]//blockquote[@id="{id}"]/div/p/text()')]
                if len(review)>=2:
                    reviews.append(' '.join(review))
                else:
                    reviews.append(review[0])

            # final dataset consolidation
            app_store_review = pd.DataFrame({'name': name, 'date': date, 'title':title, 'review': reviews, 'star': star})
            app_store_review['name'] = [n.strip() for n in app_store_review.name]
            app_store_review['date'] = [dt.strptime(d, '%d/%m/%Y').date() for d in app_store_review.date]
            app_store_review['title'] = [t.strip() for t in app_store_review.title]
            app_store_review['star'] = [int(re.findall('(\d) out of 5', s)[0]) for s in app_store_review.star]
            app_store_review.sort_values(by='date', ascending=False, inplace=True)
            app_store_review.reset_index(drop=True, inplace=True)
        return app_store_review
