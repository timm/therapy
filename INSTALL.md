# Install

usage="sh INSTALL.md"  

Ell=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )


mac=1  
unix=0 
tricks=1  

if [ "$mac" -eq 1 ]; then
  brew install bat
fi

if [ "$unix" -eq 1 ]; then
  sudo apt install bat
fi

if [ "$mac" -eq 1 -o "$unix" -eq 1 ]; then  
  sudo pip3 install docopt  #required  
  sudo pip3 install rerun pdoc3 # optional  
fi   

[ "$tricks" -eq 0 ] && exit 0

echo ""
echo "Installing tricks..."
echo ""

want=$Ell/.travis.yml
[ -f "$want" ] || cat<<'EOF'>$want
language: python
python:
  - "3.8"
cache: pip
install:
  - pip3 install -r requirements.txt
script:
  - pwd
  - ls
  - python3 -m bnbad -T
EOF

Var=$Ell/.var
mkdir -p $Var
 
want=$Var/banner
echo "Ensuring $want exists..."
[ -f "$want" ] || cat<<'EOF'>$want
EOF

want=$Var/bashrc
echo "Ensuring $want exists..."
[ -f "$want" ] || cat<<'EOF'>$want
f=$Ell/therepy/there  
alias ok="pytest  -s -k "
alias spy="rerun 'pytest $f.py'"    
ok1() { cd $Ell; pytest -s -k $1 therepy;  }  

alias gg="git pull"   
alias gs="git status"   
alias gp="git commit -am 'saving'; git push; git status"    
alias doc="sh $Ell/DOC.md; "

matrix() { nice -20 cmatrix -b -C cyan;   }
reload() { rm $Ell/.var/*; sh $Ell/INSTALL.md; . $Ell/.var/bashrc;     }
mims()   { vim -u $Ell/.var/vimrc +PluginInstall +qall; }

alias vi="vim    -u $Ell/.var/vimrc"
alias tmux="tmux -f $Ell/.var/tmuxrc"

here()  { cd $1; basename `pwd`; }    

PROMPT_COMMAND='echo -ne "🔆 $(git branch 2>/dev/null | grep '^*' | colrm 1 2):";PS1="$(here ..)/$(here .):\!\e[m ▶ "'     
EOF


want=$Var/vimrc;
echo "Ensuring $want exists..."
[ -f "$want" ] || cat<<'EOF'>$want
set list
set listchars=tab:>-
set backupdir-=.
set backupdir^=~/tmp,/tmp
set nocompatible
"filetype plugin indent on
set modelines=3
set scrolloff=3
set autoindent
set hidden "remember ls
set wildmenu
set wildmode=list:longest
set visualbell
set ttyfast
set backspace=indent,eol,start
set laststatus=2
set splitbelow
set paste
set mouse=a
set title
set number
set relativenumber
autocmd BufEnter * cd %:p:h
set showmatch
set matchtime=15
set background=light
set syntax=on
syntax enable
set ignorecase
set incsearch
set smartcase
set showmatch
set hlsearch
set nofoldenable    " disable folding
set ruler
set laststatus=2
set statusline=
set statusline+=%F
set statusline+=\ 
set statusline+=%m
set statusline+=%=
set statusline+=%y
set statusline+=\ 
set statusline+=%c
set statusline+=:
set statusline+=%l
set statusline+=\ 
set lispwords+=defthing
set lispwords+=doitems
set lispwords+=deftest
set lispwords+=defkeep
set lispwords+=labels
set lispwords+=labels
set lispwords+=doread
set lispwords+=while
set lispwords+=until
set path+=../**
if has("mouse_sgr")
    set ttymouse=sgr
else
    set ttymouse=xterm2
end
colorscheme default
set termguicolors
let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
map Z 1z=
set spell spelllang=en_us
set spellsuggest=fast,20 "Don't show too much suggestion for spell check
nn <F7> :setlocal spell! spell?<CR>
let g:vim_markdown_fenced_languages = ['awk=awk']
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'tpope/vim-fugitive'
Plugin 'scrooloose/nerdtree'
Plugin 'tbastos/vim-lua'
Plugin 'airblade/vim-gitgutter'
"Plugin 'itchyny/lightline.vim'
Plugin 'junegunn/fzf'
"  Plugin 'humiaozuzu/tabbar'
"  Plugin 'drmingdrmer/vim-tabbar'
Plugin 'tomtom/tcomment_vim'
Plugin 'ap/vim-buftabline'
Plugin 'junegunn/fzf.vim'
Plugin 'jnurmine/Zenburn'
Plugin 'altercation/vim-colors-solarized'
Plugin 'nvie/vim-flake8'
Plugin 'seebi/dircolors-solarized'
Plugin 'nequo/vim-allomancer'
Plugin 'nanotech/jellybeans.vim'
Plugin 'tell-k/vim-autopep8'
Plugin 'vimwiki/vimwiki'
Plugin 'kchmck/vim-coffee-script'
Plugin 'tpope/vim-markdown'
" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
let g:autopep8_indent_size=2
let g:autopep8_max_line_length=70
let g:autopep8_on_save = 1
let g:autopep8_disable_show_diff=1
autocmd FileType python noremap <buffer> <F8> :call Autopep8()<CR>
colorscheme jellybeans
map <C-o> :NERDTreeToggle<CR>
nnoremap <Leader><space> :noh<cr>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
set titlestring=%{expand(\"%:p:h\")}
hi Normal guibg=NONE ctermbg=NONE
hi NonText guibg=NONE ctermbg=NONE
        set fillchars=vert:\|
hi VertSplit cterm=NONE
set ts=2
set sw=2
set sts=2
 set et
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTreeType") && b:NERDTreeType == "primary") | q | endif
set hidden
nnoremap <C-N> :bnext<CR>
nnoremap <C-P> :bprev<CR>
set formatoptions-=t
set nowrap
" Markdown
let g:markdown_fenced_languages = ['awk','py=python']
EOF

if [ ! -d "$HOME/.vim/bundle" ]; then
   git clone https://github.com/VundleVim/Vundle.vim.git \
         ~/.vim/bundle/Vundle.vim
   vim -u $Var/.vimrc  +PluginInstall +qall
fi

want=$Ell/.gitignore
echo "Ensuring $want exists..."
[ -f "$want" ] || cat<<'EOF'>$want
.var
## VIM ###

# Swap
[._]*.s[a-v][a-z]
!*.svg  # comment out if you don't need vector files
[._]*.sw[a-p]
[._]s[a-rt-v][a-z]
[._]ss[a-gi-z]
[._]sw[a-p]

# Session
Session.vim
Sessionx.vim

# Temporary
.netrwhist
*~
# Auto-generated tag files
tags
# Persistent undo
[._]*.un~

### Macos ###

# General
.DS_Store
.AppleDouble
.LSOverride

# Icon must end with two \r
Icon

# Thumbnails
._*

# Files that might appear in the root of a volume
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent

# Directories potentially created on remote AFP share
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

# Python
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/
EOF

want=$Var/tmuxrc
echo "Ensuring $want exists..."
[ -f "$want" ] ||  cat<<'EOF'> $want
set -g aggressive-resize on
unbind C-b
set -g prefix C-Space
bind C-Space send-prefix
set -g base-index 1
# start with pane 1
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %
# open new windows in the current path
bind c new-window -c "#{pane_current_path}"
# reload config file
bind r source-file $Tnix/.config/dottmux
unbind p
bind p previous-window
# shorten command delay
set -sg escape-time 1
# don't rename windows automatically
set-option -g allow-rename off
# mouse control (clickable windows, panes, resizable panes)
set -g mouse on
# Use Alt-arrow keys without prefix key to switch panes
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D
# enable vi mode keys
set-window-option -g mode-keys vi
# set default terminal mode to 256 colors
set -g default-terminal "screen-256color"
bind-key u capture-pane \;\
    save-buffer /tmp/tmux-buffer \;\
    split-window -l 10 "urlview /tmp/tmux-buffer"
bind P paste-buffer
bind-key -T copy-mode-vi v send-keys -X begin-selection
bind-key -T copy-mode-vi y send-keys -X copy-selection
bind-key -T copy-mode-vi r send-keys -X rectangle-toggle
# loud or quiet?
set-option -g visual-activity off
set-option -g visual-bell off
set-option -g visual-silence off
set-window-option -g monitor-activity off
set-option -g bell-action none
#  modes
setw -g clock-mode-colour colour5
# panes
# statusbar
set -g status-position top
set -g status-justify left
set -g status-bg colour232
set -g status-fg colour137
###set -g status-attr dim
set -g status-left ''
set -g status-right '#{?window_zoomed_flag,🔍,} #[fg=colour255,bold]#H #[fg=colour255,bg=colour19,bold] %b %d #[fg=colour255,bg=colour8,bold] %H:%M '
set -g status-right '#{?window_zoomed_flag,🔍,} #[fg=colour255,bold]#H '
set -g status-right-length 50
set -g status-left-length 20
setw -g window-status-current-format ' #I#[fg=colour249]:#[fg=colour255]#W#[fg=colour249]#F '
setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '
# messages
# layouts
bind S source-file $Tnix/.config/tmux-session1
setw -g monitor-activity on
set -g visual-activity on
EOF
 
