import cupy as np
import math
import random
import click
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

ml_folder = "./ml-1m/"
TrainingTargetTitle = "ML-1M"

def matrix_factorization(R, valid_R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    logger.info('training')
    rmse_lst = []
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T
    progress = tqdm(range(steps))
    for step in progress:
        
        e_table = np.subtract(R, np.dot(P, Q))
        
        e_table = np.where(R == 0, 0, e_table)

        #print(e_table)
        p_minus_rate = np.count_nonzero(R, axis=1) * alpha * beta
        p_minus_rate = 1 - p_minus_rate
        nP = P
        nP = P * p_minus_rate.reshape(len(p_minus_rate), 1)
        nP = np.add(nP, (np.dot(e_table, Q.T))*(2*alpha))

        q_minus_rate = np.count_nonzero(R, axis=0) * alpha * beta
        q_minus_rate = 1 - q_minus_rate
        Q *= q_minus_rate
        Q = np.add(Q, (np.dot(P.T, e_table))*(2*alpha))
        P = nP
        '''
        for idx in tqdm(range(len(rates_lst))):
            i, j = rates_lst[idx]
            if R[i][j] > 0:
                # calculate error
                #eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                eij = e_table[i][j]
                for k in range(K):
                    # calculate gradient with a and beta parameter
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        #eR = np.dot(P,Q)

        e = 0
        
        for idx in tqdm(range(len(rates_lst))):
            i, j = rates_lst[idx]
            if R[i][j] > 0:
                e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                for k in range(K):
                    e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        '''
        e_table = np.subtract(valid_R, np.dot(P, Q))
        
        e_table = np.where(valid_R == 0, 0, e_table**2)
        e = np.sum(e_table)
        e_num = np.count_nonzero(e_table)
        progress.set_description("RMSE = {L:.2f}".format(L=math.sqrt(e/e_num)))

        rmse_lst.append(math.sqrt(e/e_num))
        
        #input()
        # 0.001: local minimum
        if e < 0.001:
            break

    plt.plot(rmse_lst)
    plt.title(TrainingTargetTitle + "\nRMSE = {L:.2f}".format(L=rmse_lst[-1]))
    plt.savefig(TrainingTargetTitle + ".png")

    return P, Q.T

def pre_process():
    logger.info('Pre Processing')
    movie_i2id = {}
    movie_id2i = {}
    user_i2id = {}
    user_id2i = {}

    MF = open(ml_folder + "movies.dat", encoding="utf-8", errors='ignore')
    M_lines = MF.readlines()
    MF.close()
    M = len(M_lines)
    UF = open(ml_folder + "users.dat", encoding="utf-8", errors='ignore')
    U_lines = UF.readlines()
    UF.close()
    N = len(U_lines)
    RF = open(ml_folder + "ratings.dat", encoding="utf-8", errors='ignore')
    R_lines = RF.readlines()
    RF.close()

    train_R = np.zeros((N, M), dtype=np.float64)
    valid_R = np.zeros((N, M), dtype=np.float64)
    
    for i in range(M):
        line = M_lines[i]
        m_id = line.split("::")[0]
        movie_i2id[i] = m_id
        movie_id2i[m_id] = i
    for i in range(N):
        line = U_lines[i]
        u_id = line.split("::")[0]
        user_i2id[i] = u_id
        user_id2i[u_id] = i

    cur_id = 1
    cur_rate = []
    for i in tqdm(range(len(R_lines))):
        line = R_lines[i]
        u_id, m_id, rate, timestamp = line.split("::")

        if cur_id != u_id or i == len(R_lines)-1:
            random.seed(0)
            random.shuffle(cur_rate)
            lc = len(cur_rate)
            # train R
            for u_i, m_i, rate in cur_rate[:int(lc*0.8)]:
                u_i = int(user_id2i[u_i])
                m_i = int(movie_id2i[m_i])
                rate = int(rate)
                train_R[u_i][m_i] = rate
            # valid R
            for u_i, m_i, rate in cur_rate[int(lc*0.8):]:
                u_i = int(user_id2i[u_i])
                m_i = int(movie_id2i[m_i])
                rate = int(rate)
                valid_R[u_i][m_i] = rate
            cur_id = u_id
            cur_rate = []

        cur_rate.append([u_id, m_id, rate])
        

    return N, M, train_R, valid_R


def main():
    N, M, R, valid_R = pre_process()
    
    K = 100
 
    np.random.seed(0)
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    nP, nQ = matrix_factorization(R, valid_R, P, Q, K, alpha=0.00002, steps=1000)

    nR = np.dot(nP, nQ.T)

    print(nR)

def main2():
    R = [

     [5,3,0,1],

     [4,0,0,1],

     [1,1,0,5],

     [1,0,0,4],

     [0,1,5,4],
    
     [2,1,3,0],

    ]
    
    R = np.array(R)
    # N: num of User
    N = len(R)
    # M: num of Movie
    M = len(R[0])
    # Num of Features
    K = 10

 
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

 

    nP, nQ = matrix_factorization(R, P, Q, K, alpha=0.01, steps=1000)

    nR = np.dot(nP, nQ.T)

    print(nR)

@click.command()
@click.option('-t', 'title', help='Training Target', type=str, required=True)
def params(title):
    global TrainingTargetTitle
    TrainingTargetTitle = title
    main()


if __name__ == '__main__':    
    params()

