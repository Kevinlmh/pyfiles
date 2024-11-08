import random

def judge_mutuality(x, y):
    while y != 0:
        x, y = y, x % y
    return x

x = random.randint(1, 100)
y = random.randint(1, 100)

if judge_mutuality(x, y) == 1:
    print(f"{x}和{y}是互质数")
else:
    print(f"{x}和{y}不是互质数")