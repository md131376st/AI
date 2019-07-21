import matplotlib.pyplot as plt


def read_input():
    with open("housing.data.txt") as file:
        input_ = list()
        for line in file:
            input_.append(list(map(float, line.split())))
    return input_


def drow_polt(input_, lable1, lable2, num1,tatalist):
    x_val=list()
    plt.ylabel(lable1)
    plt.xlabel(lable2)
    y_val = [x[num1] for x in input_]
    for i in range(len(input_)):
        x_val.append( round( htata(input_, i, tatalist),2))
    # plt.plot(x_val, y_val, 'ro')
    # plt.plot([0, max(y_val)], [0 + tata0, max(x_val) * tata1 + tata0], color='b', linestyle='-', linewidth=2)
    print( x_val)
    print ( y_val)
    plt.plot(x_val,y_val,'ro')
    plt.tight_layout()
    plt.show()
    pass


def cost_function(input_ , tatalist):
    cost=0
    for i in range(len(input_)):
        cost += (1/2*len(input_))*pow(htata(input_, i, tatalist)-input_[13],2)



def multivariate_linear_regression(input_):
    learning_rate = 0.000001
    tata_list = [0.0 for i in range(14)]
    # print(tata_list)
    iter_num=len(input_)
    for i in range(iter_num):
        tata_list=gradient(tata_list, input_, learning_rate)
        # print(tata_list)
    return tata_list


def htata(input_, plase, tata_list_gtad):
    sum = 0
    for i in range(len(input_[plase])):
        if i == 0:
            sum += tata_list_gtad[i]
        else:
            sum += tata_list_gtad[i]*input_[plase][i-1]
    return sum


def sub(tata_list_gtad, tata_list, learning_rate):
    for i in range(len(tata_list)):
        # print(i , ta)
        tata_list_gtad[i] = tata_list[i]-(learning_rate * tata_list_gtad[i])
    return tata_list_gtad


def gradient(tata_list, input_, learning_rate):
    tata_list_gtad = [0.0 for i in range(14)]
    # list(map(float, tata_list_gtad))
    # print(type(tata_list_gtad[0]))
    for i in range(len(input_)):
        for j in range(len(input_[0])):
            if j !=0:
                x = input_[i][j-1]
            y = input_[i][13]
            if j==0:
                tata_list_gtad[j] += (1/len(input_))*(htata(input_, i, tata_list)-y)
            else:
                tata_list_gtad[j] += (1 / len(input_)) * (htata(input_, i, tata_list) - y)*x

    return sub(tata_list_gtad, tata_list, learning_rate)


if __name__ == '__main__':
    input_ = read_input()
    tatalist=multivariate_linear_regression(input_)
    drow_polt(input_,"cost","perdiction", 13, tatalist)

