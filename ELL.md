# SHELL Tricks

usage="sh ELL.md" 

Ell=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )

echo  "$(tput bold) $(tput setaf 4) "; cat <<'EOF'
      .---.       .---.
      \_,(o,      \_,(o,
        |\_(       .)\_(        Welcome
        //_  __,   //_  __       to
       /',\\/ //  ('.\\ )'\\      (sh)ELL
       \+/ \|      \+/\/
       .+=         .+=
       _+_         _+_
      'o:o\       'o:o\
      '--'(.      '--'(,
       )   |       \    \
      '>   >        >    >
      /   /        (      \
     /   /          |     |
    o:..o;...      o:...  o:...
EOF
echo   "$(tput sgr0)"

Ell="$Ell" bash --init-file $Ell/.var/bashrc -i
