# Cycraft-Interview-Homework
This project is for 2024 Cycraft Interview Homework.
## Practices
1. Implement a TinyVectorDB that can insert and search documents.
2. Implement a python script to parse 30 news from [ITHome](https://www.ithome.com.tw/news).

## 注意
為了使腳本運行正常，請先進入python執行
```
import nltk
nltk.download('punkt')
```

## Start
其中包含4個檔案，分別是 crawler.py, tinydb.py, insert.py, search.py 其中 tinydb.py 為簡易資料庫，其他三個為功能腳本。
請依下列順序運行:

```
python3 crawler.py
python3 insert.py
python3 search.py --query-sentence '查詢內容' --limit 查詢數量 --time
```
```crawler.py``` 為爬蟲腳本，從[ITHome](https://www.ithome.com.tw/news)自動爬取30篇文章並命名```"文章標題".txt```存在news資料夾中

```insert.py``` 會將"news"資料夾內的txt檔案內容以 [bec-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)為embedding model轉換成vectort後新增至資料庫，並建立index；資料庫將以```data.json```儲存，index將以```index.bin```儲存
```search.py``` 會將```查詢內容```embedding成vector後與資料庫中的vector進行比對，並依照```查詢數量```列出最接近的vector所對應到的document內容與對應的相似分數，當輸入```--time```時，則會顯示採用hnsw搜尋以及暴力搜索所需要的時間

```python3 search.py --query-sentence '測試' --limit 2 --time```
在30筆資料中搜索的執行結果如下:
`
Use hnsw search:0.44066800
`
`
Use loop search:2.22072110
`
在僅30筆資料中進行搜索速度就差距將近5倍，若採用loop的方在更龐大的資料中檢索將花費更大量的時間成本

## Discussion Instructions

* ##### 當我們的使用情境不止需要 parse 30 篇文章時，要如何快速的 scale up 整個 parsing process？

    也許可以採用Scrapy+Scrapy-Redos進行分佈式爬蟲，加上Scrapy本身支持asynchronous ，可以同時處理大量 Request 減少等待時間。

* ##### 當我們的 DB 當中有上千萬上億筆 documents 時，要如何更近一步提升 vector database 的搜索效率？

    1. 也許可以使用NN或是其他方式進一步對Vector進行降維，降低運算量。
    2. 採用GPU運算也是一個加快搜索速度的方式。
    3. 將資料庫分區，採用平行化搜索不同的區域。