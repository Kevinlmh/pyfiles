def input_data():
    students = []
    for i in range(5):
        student_id = input(f"请输入第 {i + 1} 个学生的学号: ")
        name = input(f"请输入第 {i + 1} 个学生的姓名: ")
        scores = []
        for j in range(3):
            score = float(input(f"请输入第 {i + 1} 个学生的第 {j + 1} 门课程成绩: "))
            scores.append(score)
        students.append([student_id, name, scores])
    return students

def output_data(students):
    print("\n学生信息记录:")
    for i, student in enumerate(students):
        student_id, name, scores = student
        print(f"学生 {i + 1}: 学号: {student_id}, 姓名: {name}, 成绩: {scores}")

students = input_data()
output_data(students)
