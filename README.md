# Topic Clustering to Hot Event Detection
We will propose a pipeline for Hot Event Detection based on Topic Clustering problem, experiment and evaluate with different clustering algorithms and improve them to find the most accurate pipeline. Then, use the evaluated model to apply to the Hot Event Detection.

# news-crawler

A news crawler for BBC News and New York Times.

## Architecture

- xxx_crawler: the executive file to crawl news.
- xxx.cfg: configurations for the crawler, including api, time range and storage path etc.
- xxx_link.py: fetch download links.
- xxx_article: extract content and some meta data of one news article.

## Usage

### BBC News

```bash
python bbc_crawler.py settings bbc.cfg
```
### New York Times

```bash
python nytimes_crawler.py nytimes.cfg
```

## Configuration

Modify `reuters.cfg`, `nytimes.cfg` and `bbc.cfg` in settings folder, the main configuration items may be `start_date`, `end_date` and `path`.



# hot event detection 
## Clustering with public dataset - Topic Clustering
### Usage

```bash
python topic_clustering.py
```




## Hot event detection for News 
### Usage
```bash
python hot_event_detection.py
```


















