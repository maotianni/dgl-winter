import requests
from lxml import etree
import json
import csv
import time
import random

# 获取网页源代码
def get_page(url):
    # 构造请求头部
    headers = {
        'USER-AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
    }
    # 发送请求，获得响应
    response = requests.get(url=url, headers=headers)
    # 获得网页源代码
    html = response.text
    # 返回网页源代码
    return html


# 解析网页源代码
def parse_page(html):
    # 构造 _Element 对象
    html_elem = etree.HTML(html)
    # 详细链接
    links = html_elem.xpath('//div[@class="hd"]/a/@href')
    titles = []
    directors = []
    attrs = []
    actors = []
    genres = []
    regions = []
    languages = []
    ratings = []
    n_of_ratings = []
    rating_per_s = []
    IMDBs = []
    for i in range(len(links)):
        movie = get_page(links[i])
        html_elem_movie = etree.HTML(movie)
        # 名称
        titles.append(html_elem_movie.xpath('//div[@id="content"]/h1/span[1]/text()')[0])
        # 导演
        directors.append(html_elem_movie.xpath(
            '//div[@class="subject clearfix"]/div[@id="info"]/span[1]/span[2]/a/text()'
        ))
        # 编剧
        attrs.append(html_elem_movie.xpath(
            '//div[@class="subject clearfix"]/div[@id="info"]/span[2]/span[2]/a/text()'
        ))
        # 演员
        actors.append(html_elem_movie.xpath(
            '//div[@class="subject clearfix"]/div[@id="info"]/span[3]/span[2]/a/text()'
        ))
        # 流派
        genres.append(html_elem_movie.xpath(
            '//div[@class="subject clearfix"]/div[@id="info"]/span[@property="v:genre"]/text()'
        ))
        # 地区 与 地区
        l = html_elem_movie.xpath('//div[@class="subject clearfix"]/div[@id="info"]/text()')
        l2 = [l[i] for i in range(len(l)) if
              l[i] not in ['\n        ', ' ', ' / ', '\n        \n        ', ' ', '\n\n']]
        regions.append(l2[0])
        languages.append(l2[1])
        # 评分
        ratings.append(html_elem_movie.xpath('//div[@class="rating_self clearfix"]/strong/text()')[0])
        # 评分人数
        n_of_ratings.append(html_elem_movie.xpath('//div[@class="rating_sum"]/a/span/text()')[0])
        # 评分比例
        rating_per_s.append(html_elem_movie.xpath('//span[@class="rating_per"]/text()'))
        # IMDB链接
        IMDBs.append(html_elem_movie.xpath('//div[@class="subject clearfix"]/div[@id="info"]/a/@href'))
    data = zip(titles, directors, attrs, actors, genres, regions, languages,
               ratings, n_of_ratings, rating_per_s, IMDBs)
    # 返回结果
    return data

# 打开文件
def openfile(fm):
    fd = None
    if fm == 'txt':
        fd = open('douban.txt','w',encoding='utf-8')
    elif fm == 'json':
        fd = open('douban.json','w',encoding='utf-8')
    elif fm == 'csv':
        fd = open('douban.csv','w',encoding='utf-8',newline='')
    return fd

# 将数据保存到文件
def save2file(fm,fd,data):
    if fm == 'txt':
        for item in data:
            fd.write('----------------------------------------\n')
            fd.write('title：' + str(item[0]) + '\n')
            fd.write('directors：' + str(item[1]) + '\n')
            fd.write('attrs：' + str(item[2]) + '\n')
            fd.write('actors：' + str(item[3]) + '\n')
            fd.write('genres：' + str(item[4]) + '\n')
            fd.write('regions：' + str(item[5]) + '\n')
            fd.write('languages：' + str(item[6]) + '\n')
            fd.write('rating：' + str(item[7]) + '\n')
            fd.write('number of ratings：' + str(item[8]) + '\n')
            fd.write('rating percentages：' + str(item[9]) + '\n')
            fd.write('IMDB：' + str(item[10]) + '\n')
    if fm == 'json':
        temp = ('titles', 'directors', 'attrs', 'actors', 'genres', 'regions', 'languages',
               'ratings', 'n_of_ratings', 'rating_per_s', 'IMDBs')
        for item in data:
            json.dump(dict(zip(temp,item)),fd,ensure_ascii=False)
    if fm == 'csv':
        writer = csv.writer(fd)
        for item in data:
            writer.writerow(item)

# 开始爬取网页
def crawl():
    url = 'https://movie.douban.com/top250?start={page}&filter='
    fm = input('请输入文件保存格式（txt、json、csv）：')
    while fm!='txt' and fm!='json' and fm!='csv':
        fm = input('输入错误，请重新输入文件保存格式（txt、json、csv）：')
    fd = openfile(fm)
    print('开始爬取')
    for page in range(0,250,25):
        print('正在爬取第 ' + str(page+1) + ' 页至第 ' + str(page+25) + ' 页......')
        html = get_page(url.format(page=str(page)))
        data = parse_page(html)
        save2file(fm,fd,data)
        time.sleep(random.random())
    fd.close()
    print('结束爬取')

if __name__ == '__main__':
    crawl()