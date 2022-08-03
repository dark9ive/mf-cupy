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
alpha = 0.0
seed = 0
lam = 0.0

def mf_bpr_update(Vu, Vi, Vj):
    negxuij = (np.dot(Vu, Vi) - np.dot(Vu, Vj)) * -1        #   Negative of (xui - xuj)
    if negxuij > 500:
        negxuij = 500
    ft = math.exp(negxuij) / (1 + math.exp(negxuij))        #   Derivative of FirstTerm

    Vu = Vu + alpha * (ft * (Vi - Vj) + lam * np.linalg.norm(Vu))
    Vi = Vu + alpha * (ft * Vu + lam * np.linalg.norm(Vi))
    Vj = Vu + alpha * (ft * (-Vu) + lam * np.linalg.norm(Vj))
    return Vu, Vi, Vj

def bpr_matrix_factorization(P, Q, pos_lst, neg_mat, neg=5):
    logger.info('training')
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    progress = tqdm(range(Epochs))
    for step in progress:

        R = np.dot(P, Q.T)

        for _ in neg:
            x1, y1 = random.choice(pos_lst)
            x2 = x1
            y2 = random.choice(neg_mat[x2])

            P[x1], Q[y1], Q[y2] = mf_bpr_update(P[x1], Q[y1], Q[y2])

    return P, Q

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
    
    nP, nQ = bpr_matrix_factorization(P, Q)

    nR = np.dot(nP, nQ.T)

    print(nR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path", help="Specify dataset path", default="./ml-1m/")
    parser.add_argument("-g", "--graph", help="Specify output graph filename and title", default="ML-1M")
    parser.add_argument("-d", "--dim", help="Set Dimension", type=int, default=10)
    parser.add_argument("-e", "--epoch", help="Set epochs to be trained", type=int, default=1000)
    parser.add_argument("-lr", "-a", "--alpha", "--learningrate", help="Set learning rate(Alpha)", type=float, default=0.00002)
    parser.add_argument("-lam", help="Set lambda", type=float, default=0.1)
    parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=0)

    args = parser.parse_args()

    ml_folder = args.path
    TrainingTargetTitle = args.graph
    K = args.dim
    Epochs = args.epoch
    alpha = args.alpha
    seed = args.seed
    lam = args.lam
    
    main()
