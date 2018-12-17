# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from bs4 import BeautifulSoup
import urllib2

f=open('demtweetscsv.txt', 'rU')
f2=open('fulldemtweets.txt','a')

#f=open('reptweetscsv.txt', 'rU')
#f2=open('fullreptweets.txt','a')

lnum=1
for line in f:
    if lnum>0: #last line fixed
        sumstring=line
        linelist=sumstring.split()
        if len(linelist)>2 and linelist[0]!="RT":
            if 'http' in linelist[-1] and '…' in linelist[-2]:
                soup = BeautifulSoup(urllib2.urlopen(linelist[-1]).read(), features="lxml")

                twit_text = [p.text for p in soup.findAll('p', {'class': 'TweetTextSize--jumbo'})]
                if twit_text:
                    sumstring=" ".join(twit_text[0].split())+'\n'


        f2.write(sumstring)
    lnum+=1
