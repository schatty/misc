set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" Coloscheme like in Atom
Plugin 'https://github.com/rakr/vim-one.git'

" File explorer
Plugin 'scrooloose/nerdtree'

" Autocompletion
Plugin 'https://github.com/ycm-core/YouCompleteMe.git'

" Syntax checker
Plugin 'https://github.com/vim-syntastic/syntastic.git'

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

set number
set tabstop=4

colorscheme one " Like in Atom One theme
set background=light 
set t_Co=256 " This line is needed to working inside tmux 

autocmd vimenter * NERDTree

"Close NERDTree if it is the last open buffer
autocmd WinEnter * call s:CloseIfOnlyNerdTreeLeft()

"" Close all open buffers on entering a window if the only
" buffer that's left is the NERDTree buffer
function! s:CloseIfOnlyNerdTreeLeft()
   if exists("t:NERDTreeBufName")
       if bufwinnr(t:NERDTreeBufName) != -1
             if winnr("$") == 1
                     q
                           endif
                               endif
                                 endif
                                 endfunction

" Focus on the opended file instead of NERDTree
autocmd VimEnter * wincmd p

" Syntax checker options
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

