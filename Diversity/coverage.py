RESULT_PATH = "/home/ismp/sda1/kaiwei/NGCF-PyTorch/NGCF/result/"
#RESULT_PATH = "/home/ismp/sda1/kaiwei/old/KaiweiPaper/NGCF/result/"

from os.path import join

item_set_20 = set()
item_set_40 = set()
item_set_60 = set()
item_set_80 = set()
item_set_100 = set()
count = 0
with open(join(RESULT_PATH, "40.txt"), 'r') as f:
    for line in f.readlines():
        temp = line[1:-2]
        s = temp.split(', ')
        for a in range(0, 20):
            count = count + 1
            item_set_20.add(s[a])
            item_set_40.add(s[a])
            item_set_60.add(s[a])
            item_set_80.add(s[a])
            item_set_100.add(s[a])
        for a in range(20, 40):
            count = count + 1
            item_set_40.add(s[a])
            item_set_60.add(s[a])
            item_set_80.add(s[a])
            item_set_100.add(s[a])
        for a in range(40, 60):
            count = count + 1
            item_set_60.add(s[a])
            item_set_80.add(s[a])
            item_set_100.add(s[a])
        for a in range(60, 80):
            count = count + 1
            item_set_80.add(s[a])
            item_set_100.add(s[a])
        for a in range(80, 100):
            count = count + 1
            item_set_100.add(s[a])
        
    print(len(item_set_20))
    print(len(item_set_40))
    print(len(item_set_60))
    print(len(item_set_80))
    print(len(item_set_100))
    print(count)