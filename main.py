import numpy as np
import math
import random
import argparse
from loguru import logger

from tqdm import tqdm
import matplotlib.pyplot as plt

ml_folder = ""
recommend_file = ""
TrainingTargetTitle = ""
K = 0
Epochs = 0
alpha = 0.0
seed = 0
lam = 0.0
neg = 0
topk = 0

def mf_bpr_update(Vu, Vi, Vj):
    negxuij = (np.dot(Vu, Vi) - np.dot(Vu, Vj)) * -1        #   Negative of (xui - xuj)
    if negxuij > 500:
        negxuij = 500
    ft = math.exp(negxuij) / (1 + math.exp(negxuij))        #   Derivative of FirstTerm

    Vu = Vu + alpha * (ft * (Vi - Vj) + lam * np.linalg.norm(Vu))
    Vi = Vi + alpha * (ft * Vu + lam * np.linalg.norm(Vi))
    Vj = Vj + alpha * (ft * (-Vu) + lam * np.linalg.norm(Vj))
    return Vu, Vi, Vj

def bpr_matrix_factorization(P, Q, pos_lst, neg_mat):
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

        for _ in range(neg):
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
    # UF = open(ml_folder + "users.dat", encoding="utf-8", errors='ignore')
    # U_lines = UF.readlines()
    # UF.close()
    # N = len(U_lines)
    RF = open(ml_folder + "ratings.dat", encoding="utf-8", errors='ignore')
    R_lines = RF.readlines()
    RF.close()
    
    logger.info('Building item hashtable')
    for i in tqdm(range(M)):
        line = M_lines[i]
        m_id = line.split("::")[0]
        movie_i2id[i] = m_id
        movie_id2i[m_id] = i
    # logger.info('Building user hashtable')
    # for i in tqdm(range(N)):
    #     line = U_lines[i]
    #     u_id = line.split("::")[0]
    #     user_i2id[i] = u_id
    #     user_id2i[u_id] = i

    logger.info('Building purchase matrix')
    buf_ht = {}
    pos_lst = []
    neg_mat = []
    u_idx = 0
    for i in tqdm(range(len(R_lines))):
        line = R_lines[i]
        u_id, m_id, rate, timestamp = line.split("::")

        if u_id not in user_id2i:
            user_id2i[u_id] = u_idx
            user_i2id[u_idx] = u_id
            u_idx += 1
        
        u_i = int(user_id2i[u_id])
        m_i = int(movie_id2i[m_id])
        pos_lst.append([u_i, m_i])
        buf_ht[(u_i, m_i)] = 1
    
    N = len(user_id2i)
    matrix = np.zeros((N, M), dtype=np.float64)

    logger.info('Converting matrix')
    for i in tqdm(range(N)):
        buf = []
        for j in range(M):
            if (i, j) not in buf_ht:
                buf.append(j)
        neg_mat.append(buf)

    return N, M, pos_lst, neg_mat, user_i2id, movie_i2id

def recommend(nR, user_i2id, movie_i2id):
    logger.info("Recommend")
    f = open(recommend_file, "w")
    f.write("")
    f.close()

    for i, r in enumerate(tqdm(nR)):
        movies = np.argpartition(r, -topk)[-topk:][::-1]
        f = open(recommend_file, "a")
        for k in range(topk):
            f.write("{userID}::{movieID}::{rank}\n".format(userID=user_i2id[i], movieID=movie_i2id[int(movies[k])], rank=k+1))
        f.close()


def main():
    N, M, pos_lst, neg_mat, user_i2id, movie_i2id = pre_process()
 
    np.random.seed(seed)
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    nP, nQ = bpr_matrix_factorization(P, Q, pos_lst, neg_mat)

    nR = np.dot(nP, nQ.T)
    recommend(nR, user_i2id, movie_i2id)

    print(nR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path", help="Specify dataset path", default="./ml-10m/")
    parser.add_argument("-g", "--graph", help="Specify output graph filename and title", default="ML-1M")
    parser.add_argument("-d", "--dim", help="Set Dimension", type=int, default=10)
    parser.add_argument("-e", "--epoch", help="Set epochs to be trained", type=int, default=1000)
    parser.add_argument("-lr", "-a", "--alpha", "--learningrate", help="Set learning rate(Alpha)", type=float, default=0.01)
    parser.add_argument("-lam", help="Set lambda", type=float, default=0.1)
    parser.add_argument("-n", "--neg", help="Set sample times per epoch", type=int, default=100)
    parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=0)
    parser.add_argument("-rcmd", "--rcmd", help="Set recommend file path", default="recommend.dat")
    parser.add_argument("-topk", "--topk", help="Set recommend top k elements", type=int, default=10)

    args = parser.parse_args()

    ml_folder = args.path
    TrainingTargetTitle = args.graph
    K = args.dim
    Epochs = args.epoch
    alpha = args.alpha
    lam = args.lam
    neg = args.neg
    seed = args.seed
    recommend_file = args.rcmd
    topk = args.topk
    
    main()
