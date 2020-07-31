# Documentation

usage="sh DOC.md"   

Ell=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)  

d=$Ell/docs   
f=$Ell/therapy/thera

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

(cd $Ell/therapy; pydoc3 thera | bat -plman )
