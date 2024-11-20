def print_triangles(rows):
    for i in range(rows):
        print(" " * i, end = "")
        print("*" * (7 - 2 * i))
print_triangles(4)
