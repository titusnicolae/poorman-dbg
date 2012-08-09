#!/usr/bin/python
import re
import sys
fin=file(sys.argv[1],'r')
fout=file(sys.argv[1]+"2",'w')
regex=re.compile(r'(.*){(.*)')
white=re.compile(r'\s+')
fault=re.compile(r'class|namespace|\"')
for (i,line) in enumerate(fin):
    res=regex.match(line)
    if res: #it has a match but the beggining could be whitespace
        if white.sub('',res.group(1))=="":
            if not fault.search(last):
                fout.write(line.rstrip()+' std::cout<<"'+str(i+1)+': '+last+'"<<std::endl;'+"\n")
            else:
                fout.write(line) 
        else:
            if not fault.search(res.group(1)):
                fout.write(line.rstrip()+' std::cout<<"'+str(i+1)+': '+res.group(1)+'"<<std::endl;'+"\n")
                last=res.group(2).strip()
            else:
                fout.write(line)
    else:
        fout.write(line)
        last=line.strip()
