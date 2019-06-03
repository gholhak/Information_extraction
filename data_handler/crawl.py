import scrapy


class Crawler():
    def __init__(self):
        pass

    def init_request(self):
        urs = 'https://www.safaryabi.com/لغات-و-اصطلاحات-سفر-و-گردشگری/'
        scrapy.Request(url=urs, callback=self.parser)

    def parser(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('the file is saved!' % filename)
