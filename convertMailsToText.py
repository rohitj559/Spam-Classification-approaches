import os
from bs4 import BeautifulSoup


# os.getcwd() - path of console working directory
# "\CSDMC2010_SPAM\TRAINING" path os training dataset
sourcePath = "C:\\Users\\Rohith\\Desktop\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\TESTING"
dirs = os.listdir(sourcePath)
destPath = "C:\\Users\\Rohith\\Desktop\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\test\\"
for fname in dirs:
    source = os.path.join(sourcePath, fname)
    f = open(source, "r", errors='ignore')
    html = f.read()
    f.close()
    soup = BeautifulSoup(html, 'html.parser')
    s = soup.get_text()
    s = s.encode('UTF-8')
    f = open(destPath + fname + ".txt", "wb")
    f.write(s)
    f.close()

