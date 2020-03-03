set nocompatible              " be iMproved, required
filetype off                  " required

" Enable line number
set number

" Enable backspace for MacOS
set backspace=indent,eol,start

" Open NERD atumatically
" autocmd vimenter * NERDTree

" To toggle NERD tree
nmap <C-n> :NERDTreeToggle<CR>

" To switch to the editor on the start
" autocmd VimEnter * NERDTree | wincmd p

" Close NERD automatically if it is the only buffer
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" Open NERD tree in every tab
" autocmd BufWinEnter * NERDTreeMirror

set pastetoggle=<F2>

filetype plugin indent on
" show existing tab with 4 spaces width
set tabstop=4
" when indenting with '>', use 4 spaces width
set shiftwidth=4
" On pressing tab, insert 4 spaces
set expandtab

" To enable synax highlight
filetype plugin on
syntax on

" To use mouse in NERDTree
set mouse=a
let g:NERDTreeMouseMode=3 

set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

" set the runtime path to include Vundle and initialize
let g:python_highlight_all = 1
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

let g:jedi#force_py_version = 3

Plugin 'scrooloose/nerdtree'
Plugin 'VundleVim/Vundle.vim'
Plugin 'vim-python/python-syntax'
Plugin 'https://github.com/joshdick/onedark.vim'
Plugin 'davidhalter/jedi-vim'
Plugin 'https://github.com/vim-syntastic/syntastic.git'
Plugin 'https://tpope.io/vim/fugitive.git'
Plugin 'airblade/vim-gitgutter'

" All of your Plugins must be added before the following line
"
call vundle#end()
filetype plugin indent on    " required


" Color theme
colorscheme onedark
