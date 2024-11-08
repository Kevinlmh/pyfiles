a = [1,3,1,3,1]
b = 2
c = 5
def fun(a):
    global b
    a[0] = a[1] = 0
    for i in a: b += i
    c = b
fun(a)
c += b
s = sum(a)
print(s,b,c)