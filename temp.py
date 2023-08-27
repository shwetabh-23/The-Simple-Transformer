
def rotateArray(a, b):
        ret = []
        for i in range(len(a)):

            if b%len(a) == 0:
                ret.append(a[i+b] if len(a) > 1 else a[0])
            else:
                if i+(b%len(a)) < len(a):
                    #print((i+(b%len(a))))
                    ret.append(a[i+ (b%len(a))])
                else:
                    #print(len(a) - (i+(b%len(a))) - 1)
                    ret.append(a[(-1)*(len(a) - (i+(b%len(a))))])

        print(ret)
            
        return ret

A = [ 44, 41, 12, 42, 71, 45, 28, 65, 75, 93, 66, 66, 37, 6, 24, 59 ]
B = 18
rotateArray(A, B)

