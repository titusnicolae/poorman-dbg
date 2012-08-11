#!/usr/bin/python
import re
import sys
fin=file(sys.argv[1],'r')
fout=file(sys.argv[1]+"2",'w')
regex=re.compile(r'(.*){(.*)')
white=re.compile(r'\s+')
fault1=re.compile(r'class|namespace')
fault2=re.compile(r'class|namespace|\"')
quote=re.compile(r'"')
for (i,line) in enumerate(fin):
    res=regex.match(line)
    if res: #it has a match but the beggining could be whitespace
        if white.sub('',res.group(1))=="":
            if not fault1.search(last):
                if i==407:  
                    print "1"
                fout.write(res.group(1)+'{ std::cout<<"'+str(i+1)+': '+quote.sub(r'\"',last.strip())+'"<<std::endl;'+res.group(2)+"\n")
            else:
                fout.write(line) 
        else:
            if not fault2.search(res.group(1)):
                if i==407:  
                    print "2"
                fout.write(res.group(1)+'{ std::cout<<"'+str(i+1)+': '+res.group(1).strip()+'"<<std::endl;'+res.group(2)+"\n")
                last=res.group(2).strip()
            else:
                fout.write(line)
    else:
        fout.write(line)
        if not white.sub('',line)=="":
            last=line.strip()



