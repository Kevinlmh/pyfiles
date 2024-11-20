def num_of_daffodils():
    for num in range(100, 1000):
        hundred = num // 100
        ten = num // 10 % 10
        one = num % 10
        if hundred**3 + ten**3 + one**3 == num:
            print(num)
num_of_daffodils()