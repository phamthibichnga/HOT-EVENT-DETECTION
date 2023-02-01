import sys

from settings.dataset_conf import DatasetConfiguration
from article.bbc_article import BBCArticleFetcher


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('please input configuration path')
        exit()
    config = DatasetConfiguration()
    config.load(sys.argv[1])

    bbc_article_fetcher = BBCArticleFetcher(config)
    bbc_article_fetcher.fetch()


# pwd :  D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master>
# Run : "D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master\bbc_crawler.py" "D:\Documents\NLP\HotEventDetection\news-crawler-master\news-crawler-master\settings\bbc.cfg"
