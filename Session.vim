let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Sandboxes/UA3SAC_gym
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +313 src/main.py
badd +33 src/settings.yaml
badd +92 src/carracing.py
badd +15 src/ai_lib/UASACNet.py
badd +11 sweep.yaml
badd +70 src/ai_lib/UASAC.py
badd +34 slurm/job2.slurm
badd +223 src/ai_lib/helpers.py
argglobal
%argdel
$argadd ./
edit src/main.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 313 - ((36 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
313
normal! 09|
if exists(':tcd') == 2 | tcd ~/Sandboxes/UA3SAC_gym | endif
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFA
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
