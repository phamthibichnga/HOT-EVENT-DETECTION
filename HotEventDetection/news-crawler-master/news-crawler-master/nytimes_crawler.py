import sys

# sys.path.insert(0, 'D:/Documents/NLP/HotEventDetection/news-crawler-master/news-crawler-master')
from settings.dataset_conf import DatasetConfiguration
from article.nytimes_article import NytimeArticleFetcher


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('please input configuration path')
        exit()
    config = DatasetConfiguration()
    config.load(sys.argv[1])

    nytime_article_fetcher = NytimeArticleFetcher(config)
    nytime_article_fetcher.fetch()


# pwd :  D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master>
# Run : "D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master\nytimes_crawler.py" "D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master\settings\nytimes.cfg"
