import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import random
from newspaper import Article

base_url = "https://www.ithome.com.tw/news?page={}"
# add user agent
user_agents = [
 "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
]

headers = {
    "user-agent":random.choice(user_agents)
}

link_list = []
num = 0
page = 1
while (num < 30):
    url = base_url.format(page)
    resp =requests.get(url, headers = headers)
    soup = BeautifulSoup(resp.content, 'html.parser')
    elem = soup.select("div.view-content")
    for e in elem:
        for i in e.select("p.title a"):
            href = i.get('href')
            if href.startswith("/news/"):
                link_list.append(href)
                num += 1
                if num == 30:
                    break
    page += 1
print("共%d篇"%num)
folder_name = "news"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
a = 1
# 用標題命名可能包含非法字元
for i in link_list:
    news_url = 'https://www.ithome.com.tw'+i
    print(news_url)
    article = Article(news_url)
    article.download()
    article.parse()
    article.nlp()
    if not os.path.exists(folder_name+str(a)+".txt"):
        filename = os.path.join(folder_name,str(a)+".txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(article.text)
    a+=1