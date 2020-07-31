# Documentation

usage="sh DOC.md"   

Ell=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)  

d=$Ell/docs   
f=$Ell/therepy/there

mkdir -p $d  
  
cat $f.py |
awk '/^"""/ {i++}     
            {s=""  
             if(i==1)
                if($0 !~ /  $/) 
                  if ($0 !~ /:$/)
                     s="  ";
              print $0 s}   
' > $$
mv $$ $f.py  

pdoc3 -o $d --template-dir $d --force --html $f.py
mv $Ell/docs/there.html $Ell/docs/index.html

(cd $Ell/therepy; pydoc3 there | bat -plman )
