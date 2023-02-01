from bs4 import BeautifulSoup

from .dlink import DownloadLinkFetcher



class BBCLinkFetcher(DownloadLinkFetcher):

    BBC_FILTERS = [
        ['programmes', 21, 31],
        ['correspondents', 26, 40],
        ['iplayer', 21, 28],
        ['radio', 21, 26],
        ['live', 27, 31],
        ['m', 7, 8],
        ['video_and_audio', 26, 41]
    ]

    def _next_api(self, base_url, current_date):
        year = current_date.year
        month = current_date.month
        day = current_date.day
        api_url = base_url.format(year=year, month=month, day=day)
        return api_url

    def _html_to_links(self, html):
        soup = BeautifulSoup(html, 'lxml')
        count = 0

        # take base_api_url to get file page source (html) include articles (api : container)
        links = list()
        elements = soup.find_all('a') # Find tab <a...> 
        for element in elements:
            link = self._format_link(element['href']) # take http strings
            if self._link_filter(link, self.BBC_FILTERS) == True:
                links.append(link)

        return list(set(links)) # filter to remove duplicate
