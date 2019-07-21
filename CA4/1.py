import matplotlib.pyplot as plt


def read_input():
    with open("housing.data.txt") as file:
        input_ = list()
        for line in file:
            input_.append(list(map(float, line.split())))
    return input_


def drow_polt(input_, lable1, lable2, num1, num2, tata0, tata1):
    x_val=list()
    y_val=list()
    plt.ylabel(lable1)
    plt.xlabel(lable2)
    y_val = [x[num1] for x in input_]
    x_val = [x[num2] for x in input_]
    plt.plot(x_val, y_val, 'ro')
    plt.plot([0, max(x_val)], [0 + tata0, max(x_val) * tata1 + tata0], color='b', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.show()
    pass


def cost_function(input_ , tata0, tata1, palce):
    cost = 0
    for i in range(len(input_)):
        cost += (1/2*len(input_))*pow(tata1 * input_[i][palce]+tata0-input_[13],2)
    return cost

def univariate_regression(input_, plase):
    learning_rate = 0.01
    tata0=0
    tata1=0
    iter_num=len(input_)
    for i in range(iter_num):
        tata0,tata1=gradient(tata0, tata1, input_, learning_rate, plase)
    return tata0, tata1


def gradient(tata0, tata1, input_, learning_rate, plase):
    tata0_grad = 0
    tata1_grad = 0
    for i in range(len(input_)):
        x = input_[i][plase]
        y = input_[i][13]
        tata0_grad += (1/len(input_))*(((tata1*x)+tata0)-y)
        tata1_grad += (1/len(input_))*x*(((tata1*x)+tata0)-y)
    new_tata0=tata0-(learning_rate*tata0_grad)
    new_tata1=tata1-(learning_rate*tata1_grad)
    # print(new_tata0, new_tata1)
    return new_tata0, new_tata1


if __name__ == '__main__':
    input_ = read_input()
    tata0, tata1 = univariate_regression(input_,0)
    drow_polt(input_, "cost", "crime", 13, 0, tata0, tata1)
    tata0, tata1 = univariate_regression(input_, 2)
    drow_polt(input_, "cost","taxt" , 13, 2, tata0, tata1)

