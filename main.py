import cupy as np
import math
import random
import argparse
from loguru import logger

from tqdm import tqdm
import matplotlib.pyplot as plt

ml_folder = ""
TrainingTargetTitle = ""
K = 0
Epochs = 0
alpha = 0
seed = 0

def matrix_factorization(R, valid_R, P, Q, beta=0.02):
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
    progress = tqdm(range(Epochs))
    for step in progress:
        
        e_table = np.subtract(R, np.dot(P, Q))
        
        e_table = np.where(R == 0, 0, e_table)

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
        
        #   Count Loss

        e_table = np.subtract(valid_R, np.dot(P, Q))
        
        e_table = np.where(valid_R == 0, 0, e_table**2)
        e = np.sum(e_table)
        e_num = np.count_nonzero(e_table)
        progress.set_description("RMSE = {L:.2f}".format(L=math.sqrt(e/e_num)))

        rmse_lst.append(math.sqrt(e/e_num))

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
    
 
    np.random.seed(seed)
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    nP, nQ = matrix_factorization(R, valid_R, P, Q)

    nR = np.dot(nP, nQ.T)

    print(nR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path", help="Specify dataset path", default="./ml-1m/")
    parser.add_argument("-g", "--graph", help="Specify output graph filename and title", default="ML-1M")
    parser.add_argument("-d", "--dim", help="Set Dimension", type=int, default=10)
    parser.add_argument("-e", "--epoch", help="Set epochs to be trained", type=int, default=1000)
    parser.add_argument("-lr", "-a", "--alpha", help="Set learning rate(Alpha)", type=float, default=0.00002)
    parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=0)

    args = parser.parse_args()

    ml_folder = args.path
    TrainingTargetTitle = args.graph
    K = args.dim
    Epochs = args.epoch
    alpha = args.alpha
    seed = args.seed
    

    main()
