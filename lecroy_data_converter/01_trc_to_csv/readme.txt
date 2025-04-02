
使い方
gcc -o ConvTrcToCSV ConvTrcToCSV.c
./ConvTrcToCSV


"../trc/C%d--wave--%05d.trc"を開こうとするので、場合に応じて修正する必要がある
int N, int chを自分の使うデータ数、最小のch番号に合わせる

trcフォルダ
csvフォルダ
を作成する

*zsh: segmentation fault  が出ることが多い
1MSに近い場合これが出るのでサンプル数を減らすことが勧められる。コードの修正も行うべきである

* cannot open the file to read




参考：オリジナルのプログラム
https://teledynelecroy.com/japan/solutions/program/sampleprogram/gcc_lecroybintoascii/default.asp

オリジナルのプログラムだとwarningが出るが,
これはFILE型を閉じるのにclose()を使っているのが間違い.
fclose()にしないといけない.
FILE型のクローズを間違っているが, メモリの許す限り(?)は動く.
しかしなまじ動くからといって使うと, たとえばループさせると途中で止まってしまう.
warningにも気を使うべきという好例か.

