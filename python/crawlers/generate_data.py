import crawl_data


class Crawler():
    def __init__(self,keyword=None,time=None):
        self.keyword=keyword
        self.time=time

    def craw_data(self):
        #  crawl data
        keyword = self.keyword
        time = self.time

        twitterData = crawl_data.TwitterData()
        tweets = twitterData.getTwitterData(keyword,time)
        print ('Done')

        # store params
        with open('data/params.txt','w') as f:
            f.write(keyword+'\n')
            f.write(time+'\n')
        return keyword,time,tweets


    def pre_csv_data(self):
        from  utils.Preprocess_Tweets import preprocess_data
        from utils.txt2csv import txt2csv
        #  Preprocess data
        preprocess_data()

        #  Prepare csv
        txt2csv()
        print 'raw csv data generated'



if __name__ =='__main__':
    # crawl data
    crawler = Crawler(keyword='iphone8 iphoneX', time='lastweek')
    keyword, time, tweets = crawler.craw_data()

    from  utils.Preprocess_Tweets import preprocess_data
    from utils.txt2csv import txt2csv
    #  Preprocess data
    preprocess_data()

    #  Prepare csv
    txt2csv()



