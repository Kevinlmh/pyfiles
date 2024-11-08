def num_to_weekday():
    weekdays = {
        1: 'Monday',
        2: 'Tuesday',
        3: 'Wednesday',
        4: 'Thursday',
        5: 'Friday',
        6: 'Saturday',
        7: 'Sunday'
    }
    num = int(input("请输入一个1到7之间的数字："))
    if 1 <= num <= 7:
        print(weekdays[num])
    else:
        print("输入无效，请输入一个1到7之间的数字")
num_to_weekday();