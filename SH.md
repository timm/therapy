# SHELL Tricks

usage=". SHELL.md" 

Ell=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )  

f=$Ell/therapy/thera  

here()  { cd $1; basename `pwd`; }    

alias gg="git pull"   
alias gs="git status"   
alias gp="git commit -am 'saving'; git push; git status"    
alias ok="pytest.py  $f.py"
alias spy="rerun 'pytest $f.py'"    
alias doc="sh $Ell/DOC.md"  

ok1() { pytest -s -k $1 $f.py;  }  

PROMPT_COMMAND='echo -ne "🔆 $(git branch 2>/dev/null | grep '^*' | colrm 1 2):";PS1="$(here ..)/$(here .):\!\e[m ▶ "'     
