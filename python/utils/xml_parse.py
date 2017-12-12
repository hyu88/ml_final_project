'''
Parse xmls from sorted_data, generate new xmls and collect them at sorted_data/full_train.txt
'''

#!/usr/bin/python
# -*- coding: UTF-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
import glob


def clean_xml(filename):

    with open(filename, 'r') as f:
        filedata = f.read()
        filedata = filedata.replace('&', '')
        filedata = re.sub(u"[^\x20-\x7f]+",u"", filedata)
        filedata = '<root>' + filedata + '</root>'

    with open(filename + '.xml', 'w') as fout:
        fout.write(filedata)



def parse_xml(filename):
    datas=[]
    try:
        DOMTree = xml.dom.minidom.parse(filename)
        collection = DOMTree.documentElement
        reviews = collection.getElementsByTagName("review")

        for review in reviews:

            texts = review.getElementsByTagName('review_text')

            for text in texts:

                if 'negative' in filename:
                    data = "\"negative\"" + ',\"' + texts[0].childNodes[0].data + "\""
                else:
                    data = "\"positive\"" + ',\"' + texts[0].childNodes[0].data + "\""

                datas.append(data)
    except:
        print "{} can't be parsed".format(filename)

    return datas




if __name__=='__main__':

    path='../sorted_data'
    dirs=os.listdir(path)[1:]
    dirs.remove('stopwords')
    dirs.remove('summary.txt')
    try:
        dirs.remove('full_train.txt')
    except:
        pass

    lines=[]
    for dir in dirs:

        negFile=os.path.join(path,dir,'negative.review')
        posFile = os.path.join(path, dir, 'positive.review')

        if not os.path.exists(negFile):
            print '{} not exist'.format(negFile)
        if not os.path.exists(posFile):
            print '{} not exist'.format(posFile)


    #     clean xml
        clean_xml(negFile)
        clean_xml(posFile)


    #     parse xml
        negXml=negFile+'.xml'
        posXml=posFile+'.xml'

        lines+=parse_xml(posXml)
        lines+=parse_xml(negXml)

    print len(lines)
    with open('../data/full_train.txt', 'a') as fout:
        for line in lines:
            fout.write('%s\n' % line)


