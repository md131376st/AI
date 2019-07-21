import random
import time


def reading_input():
    d,t= map(int, raw_input().split())
    c =int(raw_input())
    Cvec=map(int, raw_input().split())
    p=int(raw_input())
    mypflist=list()
    sadlist=list()
    for x in xrange(p):
        pp=map(int, raw_input().split())
        mypflist.append(pp)
        pass
    for y in xrange(c):
        sadlist.append( map(int, raw_input().split()))
    return d,t,c,p,Cvec,mypflist,sadlist
    pass


working_cor = list()
Cvec = list()


def randomgenerator(mypflist, p):#dont forget genaret d*t*10
    #works correct
    for x in xrange(10*d*t):
        coromos = list()
        pc=set()
        for y in xrange(d):
            for z in xrange(t):
                listgin=list()
                listgin.append(y)
                listgin.append(z)
                pc_=genaratgin(mypflist, p,pc)
                if pc_==None:
                   pc_=[]
                listgin.append(pc_)
                coromos.append(listgin)
        # print len(coromos)
        working_cor.append(coromos)


def genaratgin(mypflist, p,pc):#works correct
    # print "hi"
    randnum = int(random.uniform(1, p+1))
    list_=list()
    proff = set()
    class_=set()
    for w in xrange(randnum):
        count = 0
        myp=int(random.uniform(0, p - 1))
        while myp in proff and count<p*5:
            count+=1
            myp = int(random.uniform(0, p - 1))
        if count>=p*5:
            continue
        count=0
        proff.add(myp)
        myc = int(random.uniform(1, mypflist[myp][0]-1))
        count=0
        # print proff,class_,p
        while( str([myp, mypflist[myp][myc]]) in pc or mypflist[myp][myc] in class_ )and count<p*5:
            # print count
            myc = int(random.uniform(1, mypflist[myp][0]-1))
            count+=1
        if count >= p * 5:
            continue
        class_.add(mypflist[myp][myc])
        pc.add(str([myp, mypflist[myp][myc]]))

        # print w,"count"
        # print proff ,"proff"
        # print pc ,"pc"
        list_.append([myp,mypflist[myp][myc]])
    return list_


def validation(d,t):
    x=0
    while x<len(working_cor):
        num = each(d,t,x)
        if num is 0:
            x+=1
    pass


def each(d, t, x):
    if len(working_cor[x]) < d*t:
        working_cor.pop(x)
        return -1
    for y in xrange(len(working_cor[x])):
        pc = set()
        class_=set()
        for z in xrange(len(working_cor[x][y][2])):
                num=str(working_cor[x][y][2][z])
                if num not in pc and working_cor[x][y][2][1] not in class_ :
                    pc.add(num)
                    class_.add(working_cor[x][y][2][1])
                else:
                    working_cor.pop(x)
                    return -1

    return 0


def fitness():
    for x in xrange(len(working_cor)):
        valuecoromozon(working_cor[x],x)
    pass


def valuecoromozon(coromozon,ind):
    # print coromozon
    benefit=0
    # print coromozon
    for x in xrange(len(coromozon)-1):
        sum=0
        for y in xrange(len(coromozon[x][2])):
            # print "line 110",coromozon[x][2][y][1]
            sum += Cvec[coromozon[x][2][y][1] - 1]
            if len(coromozon[x][2])==1:
                break
            else:
                for w in xrange(y, len(coromozon[x][2]), 1):
                    sum -= sadlist[y][w]
        if len(working_cor[ind][x])==3:
            working_cor[ind][x].append(sum)
        else:
            working_cor[ind][x][3]=sum
        benefit+=sum

    if len(working_cor[ind])==d*t:
        working_cor[ind].append(benefit)
    else:
        working_cor[ind][d*t]=benefit


def miotation(corn,bool_=True):
    pc = set()
    class_=set()
    for i in xrange(len(corn) - 1):
        for j in xrange(len(corn[i][2])):
            if str(corn[i][2][j]) in pc:
                corn[i][2].pop(j)
                return corn
            if corn[i][2][j][1] in class_:
                # print corn
                # print corn[i][2][j]
                corn[i][2].pop(j)
                # print corn[i][2],"??????"
                return corn
            pc.add(str(corn[i][2][j]))
            class_.add(corn[i][2][j][1])
    if bool_==True:
        for i in xrange(len(corn)-1):#each
            p=set()

            for j in xrange(len(corn[i][2])):
                if corn[i][2][j][0] not in p:
                    p.add(corn[i][2][j][0])
            for j in xrange(len(mypflist)):
                if j not in p:
                    for z in xrange(len(mypflist[j])):

                        if str([j,mypflist[j][z]]) not in pc and mypflist[j][z] not in class_:
                            corn[i][2].append([j,mypflist[j][z]])
                            pc.add(str([j,mypflist[j][z]]))
                            class_.add(mypflist[j][z])
                            p.add(j)
                            break
    else:
        for i in xrange(len(corn) - 1):
            # print corn
            if len(corn[i][2])==0:
                # print "hi"
                myp=int(random.uniform(0,len(mypflist)-1))
                for j in xrange(len (mypflist[myp])):
                    if mypflist[myp][j] not in class_:
                        corn[i][2].append([myp, mypflist[myp][j]])
                        # print "hi"
                        pc.add(str([myp, mypflist[myp][j]]))
                        class_.add(mypflist[myp][j])
                break

            j = int(random.uniform(0,len(corn[i][2])-1))
            ind = int(random.uniform(0, j))
            pc.remove(str(corn[i][2][ind]))
            # print corn[i][2]
            # print corn[i][2][ind]
            # print corn[i][2][ind][1]
            # print class_
            class_.remove(corn[i][2][ind][1])
            corn[i][2].pop(ind)


    return corn


def validcross(d, t, corn):
    pc = set()
    for x in xrange(d*t):
        z = 0
        while z < (len(corn[x][2])):
                num = str(corn[x][2][z])
                if num not in pc:
                    pc.add(num)
                    z += 1
                else:
                    corn[x][2].pop(z)


    # print "line 156"
    # print corn


def cross_over(d, t):
    num = 0
    role= 0
    while num < d*t*2:
        if role==0:
            sel1 = int(random.uniform(0, d*t))
            sel2 = int(random.uniform(0, d*t))
            role=1
        else:
            sel1 = int(random.uniform(0, d * t*10))
            sel2 = int(random.uniform(0, d * t*10))
            role=0
        while sel1 == sel2:
            sel2 = int(random.uniform(0, d * t))
        pos = int(random.uniform(0, d*t-1))
        corn1 = working_cor[sel1][:pos]+working_cor[sel2][pos:len(working_cor[sel2])-1]
        validcross(d, t, corn1)
        working_cor.append(corn1)
        valuecoromozon(corn1, len(working_cor)-1)
        corn2=working_cor[sel2][:pos]+working_cor[sel1][pos:len(working_cor[sel1])-1]
        validcross(d, t, corn2)
        working_cor.append(corn2)
        valuecoromozon(corn2, len(working_cor) - 1)
        num += 1
    pass

start=time.time()
d, t, c, p, Cvec, mypflist, sadlist = reading_input()
randomgenerator(mypflist, p)
fitness()
working_cor.sort(key=lambda x: x[d * t], reverse=True)
# print working_cor
# for x in xrange(len( working_cor)):
#
#     print working_cor[x][d*t],"/////////////////"
num=0
for x in xrange(10000):
    if time.time() - start >110:
        # print time.time() - start
        break
    cross_over(d, t)
    working_cor.sort(key=lambda x: x[d * t], reverse=True)
    for y in xrange(d*t):
        sel = int(random.uniform(0, d * t*11-1))
        mybool=True
        if working_cor[0][len(working_cor[0])-1]<0:
            mybool=False
        if y%7==0:
            mybool=False
        working_cor[sel] = miotation(working_cor[sel],mybool)
        valuecoromozon(working_cor[sel], sel)
    working_cor = working_cor[:d * t * 10]

working_cor.sort(key=lambda x: x[d * t], reverse=True)
print working_cor[0][len(working_cor[0])-1]
for x in xrange(d*t):
    print working_cor[0][x][0],working_cor[0][x][1],working_cor[0][x][2]

# print len(working_cor)
# for x in xrange(len(working_cor)):
#     print working_cor[x][d * t]











