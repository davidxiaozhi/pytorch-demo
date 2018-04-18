#!/usr/bin/env python
# -*- coding:UTF-8 -*-

if __name__ == '__main__':
    file_name = "/Users/lizhipeng/Downloads/周杰伦歌词大全.txt"
    with open(file_name ,'r',encoding="gb18030") as f:
        with open("/export/temp/周杰伦歌词大全.txt", 'w', encoding="UTF-8") as fw:
            #print(f.readlines())
            fw.writelines(f.readlines())