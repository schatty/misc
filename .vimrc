set number

" Plugings will be downloaded under the specified directory
call plug#begin('~/.vim/plugged')

" Declare the list of plugins
Plug 'tpope/vim-sensible'
Plug 'junegunn/seoul256.vim'
Plug 'scrooloose/nerdtree'
Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }

" List ends here. Plugins become visible to Vim after this call.
call plug#end()

let g:seoul256_background = 256
colo seoul256-light

" Launch NERDTree at launch
autocmd vimenter * NERDTree

" Focus on the opning file when vim opens
autocmd vimenter * wincmd p

" Close all open windows on exit
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
