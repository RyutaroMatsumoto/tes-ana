使い方

これは時間情報をカットして波形をnpyに格納するものである
時間間隔はdt.txtに出力される
対象ファイル名は
directory = "../csv/"
file_prefix = "C3--noise--"
file_suffix = ".csv"
が指定されているため、必要に応じて変更する必要がある。


Pythonの実行コードは以下

conda deactivate
source /Users/ryutaro_matsumoto/Desktop/Reaserch/Analysis_codes/myenv/bin/activate
Python 00_csv2npy.py


よくあるエラー

Traceback (most recent call last):
  File "00_csv2npy.py", line 34, in <module>
    pulse[i, :] = data[0:dp, 1]
ValueError: could not broadcast input array from shape (1001,) into shape (1002,)

データの数が揃っていないといけない
ファイル中のdpの値をできる範囲で大きく設定すること。
ファイル数(n)も揃っている必要がある。
