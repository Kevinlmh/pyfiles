def fib(n):
    if n >= 2:
        return fib(n-1)+fib(n-2)
    else:
        return 1
lista=[fib(n) for n in range(10)]
print(lista)
print(*lista[3:6])