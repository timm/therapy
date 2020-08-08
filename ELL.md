# SHELL Tricks

usage="sh ELL.md" 

Ell=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )

tput bold; tput setaf 4
cat <<'EOF'
   .-.
  (o o)  boo !!
  | O \    there's no escape 
   \   \     from (sh)ELL, v0.4
    `~~~'
EOF
tput sgr0

Ell="$Ell" bash --init-file $Ell/.var/bashrc -i
