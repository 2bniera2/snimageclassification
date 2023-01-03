


matrix =[
    [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ],
    [8 ,9 ,10,11,12,13,14,15],
    [16,17,18,19,20,21,22,23],
    [24,25,26,27,28,29,30,31],
    [32,33,34,35,36,37,38,39],
    [40,41,42,43,44,45,46,47],
    [48,49,50,51,52,53,54,55],
    [56,57,58,59,60,61,62,63]

]

def zz(m):
    lis = []
    i, j = 0, 0
    mode = "right"
    c = 0
    while c < 63:
        
        # first half
        if mode == "right":
            j+=1
            print(m[i][j])
            c+=1
            mode = "leftdown"
        if mode == "leftdown":
            while (j > 0 and i < 7):
                j-=1
                i+=1
                print(m[i][j])
                c+=1
            mode = "down"
        if mode == "down":
            # print(f"i: {i} j: {j}")
            if i < 7 and j < 7:
                i+=1
                print(m[i][j])
                c+=1
                mode = "rightup"
            elif i == 7: 
                j += 1
                c += 1
                print(m[i][j])
                mode = "rightup"
            elif j == 7:
                i += 1
                c +=1 
                print(m[i][j])
                mode = "leftdown"
            # print(mode)
        if mode == "rightup":
            while (j < 7 and i > 0):
                j+=1
                i-=1
                c+=1
                print(m[i][j])
            if j < 7:
                mode = "right"
            else: 
                mode = "down"
        
zz(matrix)
            


            

