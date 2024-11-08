a = [-1,4]
for i in range(3):
    try:
        assert i>0
        assert a[i]>=0
        print(a[i]**0.5,end=" ")
    except AssertionError: print("AE",end=" ")
    except IndexError: print("IE",end=" ")
    except: print("UnknownError",end=" ")