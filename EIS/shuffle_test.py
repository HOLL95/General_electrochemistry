import random
z=["a", "b", "c", "d", "e", "f", "g", "h"]
a=list(enumerate(z))
random.shuffle(a)
winners=[0]*int(len(z)/2)
for i in range(0, len(a), 2):
    winner=random.randint(0, 2)
    if winner==0:
        win_idx=i
    else:
        win_idx=i+1
    print(a[i], a[i+1], "->", a[win_idx])
    winners[int(i/2)]=a[win_idx][0]
print([z[x] for x in winners])
