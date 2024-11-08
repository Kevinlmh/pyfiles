def merge_files(file_a, file_b, file_c):
    with open(file_a, 'r') as a:
        data_a = a.read().strip()
    with open(file_b, 'r') as b:
        data_b = b.read().strip()

    merged_data = sorted(data_a + data_b)

    with open(file_c, 'w') as c:
        c.write(''.join(merged_data))

merge_files('A.txt', 'B.txt', 'C.txt')
