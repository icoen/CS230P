""" text preprocessing """

import fileinput
import re
import json
import math
import sys


FLAGS = re.MULTILINE | re.DOTALL
def re_sub(pattern, repl, sumstring):
    return re.sub(pattern, repl, sumstring, flags=FLAGS)

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body) #lower case
    else:
        result = " ".join(re.sub( r"([A-Z])", r" \1", hashtag_body, flags=FLAGS).split()) #if a mix, split at capital letters
    return result        

def textpre(f,f2):
    for line in f:
        sumstring=line
        linelist=sumstring.split()
        sumstring=' '.join(word for word in sumstring.split(' ') if not word.startswith('http')) #removes words that starts with http
        sumstring = re_sub(r"pic.twitter.com\/\S*", "", sumstring) #remove picture links
        sumstring = re_sub(r"https?:\/\/\S+|www\.(\w+\.)+\S*", "", sumstring) #remove residual http or www links

        sumstring = re_sub(r"#\S+", hashtag, sumstring) #if hashtag process hashtag
        sumstring = re.sub("&amp;", "&", sumstring)  
        sumstring = re.sub("[(),!?\'`\":;.+@#$]", " ", sumstring) #remove all punctuation and insert a space in its place

        sumstring=' '.join(word for word in sumstring.split()) #removes consecutive empty space
        if sumstring!="\n" :  #not empty
            f2.write(sumstring.lower()+'\n') #add back the newline that we stripped

f=open('fullreptweets.txt', 'rU')
f2=open('twtfullrepn.txt','w')
textpre(f,f2)

f3=open('fulldemtweets.txt', 'rU')
f4=open('twtfulldemn.txt','w')
textpre(f3,f4)

f.close()
f2.close()
f3.close()
f4.close()
