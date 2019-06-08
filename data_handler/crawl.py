import scrapy


class Crawler(scrapy.Spider):
    def __init__(self):
        self.name = "quotes"

    def init_request(self):
        urs = ['http://quotes.toscrape.com/page/1/',
               'http://quotes.toscrape.com/page/2/',
               ]

        scrapy.Request(url=urs, callback=self.parser)

    def parser(self, response):
        page = response.url.split("/")[-2]
        filename = 'data_handler/quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('the file is saved!' % filename)
