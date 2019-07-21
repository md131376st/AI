import matplotlib.pyplot as plt
import numpy as np
import math

learning_rate=0.05

def sigmonid(tataTx):
    # print(1/1+np.exp(- tataTx),"!!!!!!!!!")
    return float( 1.0/float(1.0+np.exp(-1.0*tataTx)))


def readinput():
    with open("logistic_and_svm_data.txt") as file:
        input_ = list()
        for line in file:
            input_.append(list(map(float, line.split(','))))
    return input_


def logistic_regression(input_,tataprive):
    tata_list_gtad = [0.0 for i in range(3)]
    for i in range(len(input_)):
        x1=input_[i][0]
        x2=input_[i][1]
        y=input_[i][2]

        # tata_list_gtad[0] += (1 / len(input_)) * (sigmonid(tatatx(x1,x2,tataprive[0],tataprive[1],tataprive[2])) - y)
        tata_list_gtad[1] += (1 / len(input_)) * (sigmonid(tatatx(x1,x2,tataprive[0],tataprive[1],tataprive[2])) - y) * x1
        tata_list_gtad[2] += (1 / len(input_)) * (sigmonid(tatatx(x1, x2,tataprive[0],tataprive[1],tataprive[2])) - y) * x2
    # print(tata\prive,tata_list_gtad)
    # tataprive[0] =tataprive[0]- learning_rate*tata_list_gtad[0]
    tataprive[1] = tataprive[1] - learning_rate*tata_list_gtad[1]
    tataprive[2] =  tataprive[2]-learning_rate * tata_list_gtad[2]
    pass

def tatatx( x1,x2, tata0,tata1,tata2):
    return tata1*x1+tata2*x2


def costfunction(input_,tataprive):
    cost=0.0
    for i in range(len(input_)):
        htata=tatatx(input_[i][0],input_[i][1],tataprive[0],tataprive[1],tataprive[2])
        # print(htata)
        if input_[i][2]==1:
            cost+=float(input_[i][2]*math.exp(htata))
            # print(input_[i][2]*math.exp(htata),"!!!!!!!!!")
        else:
            cost+=float((1-input_[i][2])*math.exp(1-htata))
            # print((1-input_[i][2])*math.exp(1-htata),"???????")
    print(cost/len(input_))
    pass


def drow_polt(input_):
    xlist_r=list()
    ylist_r=list()
    xlist_g = list()
    ylist_g = list()
    for i in range(len(input_)):
        if input_[i][2]==0.0:
            xlist_r.append(input_[i][0])
            ylist_r.append(input_[i][1])
        else:
            xlist_g.append(input_[i][0])
            ylist_g.append(input_[i][1])

    plt.scatter(xlist_r,ylist_r,color='red')
    plt.scatter(xlist_g, ylist_g, color='green')
    # plt.show()


if __name__ == '__main__':

    list_ = readinput()
    input_ = np.array(list_)
    drow_polt(input_)
    tataprive = [0.0 for i in range(3)]
    for i in range(100):
        logistic_regression(input_,tataprive)
        costfunction(input_, tataprive)
    y=list()

    print(tataprive)

    plt.plot([0, max(input_[:,0])],[tataprive[1]* max(input_[:,0])+tataprive[2]*max(input_[:,1]),0],color='b', linestyle='-', linewidth=2)

    # plt.scatter(,y)
    plt.show()
    #
    # # print(input_)