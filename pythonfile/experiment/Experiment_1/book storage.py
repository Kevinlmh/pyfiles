class Book:
    def __init__(self, name, publisher, price, count):
        self.name = name
        self.publisher = publisher
        self.price = float(price)
        self.count = int(count)

def loadBook(booklist):
    with open('books.txt', 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 4:
                name, publisher, price, count = data
                booklist.append(Book(name, publisher, price, count))

def sortBook(booklist):
    booklist.sort(key=lambda book: (book.publisher, book.count))

def saveBook(booklist):
    with open('sorted_books.txt', 'w') as file:
        for book in booklist:
            file.write(f"{book.name} {book.publisher} {book.price:.2f} {book.count}\n")

def main():
    booklist = []
    loadBook(booklist)
    sortBook(booklist)
    saveBook(booklist)

main()
