let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Sandboxes/e2SAC
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd ~/Sandboxes/e2SAC/
edit src/main.py
argglobal
balt src/settings.yaml
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 205 - ((22 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 205
normal! 049|
if exists(':tcd') == 2 | tcd ~/Sandboxes/e2SAC | endif
tabnext 1
badd +147 ~/Sandboxes/e2SAC/e2SAC/UASAC.py
badd +1 ~/Sandboxes/e2SAC/src/mainSAC.py
badd +205 ~/Sandboxes/e2SAC/src/main.py
badd +43 ~/Sandboxes/e2SAC/utils/replay_buffer.py
badd +6 ~/Sandboxes/e2SAC/src/settings.yaml
badd +54 ~/Sandboxes/e2SAC/src/carracing.py
badd +124 ~/Sandboxes/e2SAC/src/shebangs.py
badd +58 ~/Sandboxes/e2SAC/e2SAC/UASACNet.py
badd +60 ~/Sandboxes/e2SAC/e2SAC/normal_inverse_gamma.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=90 shortmess=filnxtToOFA
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
