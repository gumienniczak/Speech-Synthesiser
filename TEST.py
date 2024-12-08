import re
date_format = re.compile(r'(\d+)/(\d+)/?(\d{2,})?')

phrase = 'I was born on 19/12/89 it was amazing 13/22/22'
c = '19/12/1992'




lkp = date_format.search(phrase)
while lkp:
    phrase.replace(lkp, 'a')
    print(lkp.group(0))
