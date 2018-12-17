#reads in the third column of the csv tweet file, removes newlines within the third column
import csv

f=open('demtweetscsv.txt', 'w')
f2=open('reptweetscsv.txt', 'w')

with open('ExtractedTweets.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
           
            line_count += 1
        else:
          
            line_count += 1
            if row[0]=='Democrat':
                f.write(" ".join(row[2].split())+'\n')
            if row[0]=='Republican':
                f2.write(" ".join(row[2].split())+'\n')

