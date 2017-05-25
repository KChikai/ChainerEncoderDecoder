# -*- coding: utf-8 -*-

import re


def main():
    pattern = r'(.*?)(\t)(.*?)(\n|\r\n)'
    r = re.compile(pattern=pattern)
    f_post = open('data/post.txt', 'w', encoding='utf-8')
    f_cmnt = open('data/cmnt.txt', 'w', encoding='utf-8')
    f_test = open('data/post-test.txt', 'w', encoding='utf-8')
    for index, row in enumerate(open('data/pair_corpus.txt', 'r', encoding='utf-8')):
        m = r.match(row)
        if m is not None:
            if index % 100 != 0:
                f_post.write(m.group(1) + '\n')
                f_cmnt.write(m.group(3) + '\n')
            else:
                f_test.write(m.group(1) + '\n')
        if index == 5000:
            break

if __name__ == '__main__':
    main()
