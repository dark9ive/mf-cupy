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
part = 0

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
    # counting total lines
    with open(ml_folder + "ratings.dat") as fp:
        for total_lines, _ in enumerate(fp):
            pass
    total_lines += 1

    Ns = []
    Ms = []
    user_i2ids = []
    movie_i2ids = []
    pos_lsts = []
    neg_mats = []

    logger.info('Building rating matrix')
    from_p, to_p, p_cnt, p_part = 0, total_lines//part, 0, 0
    with open(ml_folder + "ratings.dat") as file:
        line = file.readline()
        while(line):
            # partition log info
            # initialize
            if p_cnt == from_p:
                logger.info('building from {} to {}, total {}'.format(from_p, to_p, total_lines))
                movie_i2id = {}
                movie_id2i = {}
                mov_idx = 0
                user_i2id = {}
                user_id2i = {}
                usr_idx = 0
                buf_ht = {}
                pos_lst = []

            u_id, m_id, rate, timestamp = line.split("::")

            if u_id not in user_id2i:
                user_id2i[u_id] = usr_idx
                user_i2id[usr_idx] = u_id
                usr_idx += 1

            if m_id not in movie_id2i:
                movie_id2i[m_id] = mov_idx
                movie_i2id[mov_idx] = m_id
                mov_idx += 1

            u_i = int(user_id2i[u_id])
            m_i = int(movie_id2i[m_id])
            pos_lst.append([u_i, m_i])
            buf_ht[(u_i, m_i)] = 1

            line = file.readline()

            # handling partition offset
            p_cnt += 1
            if p_cnt == to_p:
                p_part += 1
                from_p = total_lines*p_part//part
                to_p = total_lines*(p_part+1)//part
    
                N = len(user_id2i)
                M = len(movie_id2i)
                Ns.append(N)
                Ms.append(M)
                user_i2ids.append(user_i2id)
                movie_i2ids.append(movie_i2id)
                pos_lsts.append(pos_lst)
                neg_mat = [[]] * N

                logger.info('Converting matrix')
                for i in tqdm(range(N)):
                    buf = []
                    for j in range(M):
                        if (i, j) not in buf_ht:
                            buf.append(j)
                    neg_mat[i] = buf
                neg_mats.append(neg_mat)

    return Ns, Ms, pos_lsts, neg_mats, user_i2ids, movie_i2ids

def recommend(nRs, user_i2ids, movie_i2ids):
    logger.info("Recommend")
    f = open(recommend_file, "w")
    f.write("")
    f.close()

    for p in range(part):
        logger.info("Recommend at part {}, total {}".format(p+1, part))
        nR = nRs[p]
        for i, r in enumerate(tqdm(nR)):
            movies = np.argpartition(r, -topk)[-topk:][::-1]
            f = open(recommend_file, "a")
            for k in range(topk):
                f.write("{userID}::{movieID}::{rank}\n".format(userID=user_i2ids[p][i], movieID=movie_i2ids[p][int(movies[k])], rank=k+1))
            f.close()


def main():
    Ns, Ms, pos_lsts, neg_mats, user_i2ids, movie_i2ids = pre_process()

    nRs = []
    for p in range(part):
        np.random.seed(seed)
        P = np.random.rand(Ns[p], K)
        Q = np.random.rand(Ms[p], K)
        
        nP, nQ = bpr_matrix_factorization(P, Q, pos_lsts[p], neg_mats[p])

        nR = np.dot(nP, nQ.T)
        nRs.append(nR)
        print(nR)

    recommend(nRs, user_i2ids, movie_i2ids)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path", help="Specify dataset path", default="./ml-10m/")
    parser.add_argument("-g", "--graph", help="Specify output graph filename and title", default="ML-1M")
    parser.add_argument("-d", "--dim", help="Set Dimension", type=int, default=10)
    parser.add_argument("-e", "--epoch", help="Set epochs to be trained", type=int, default=500)
    parser.add_argument("-lr", "-a", "--alpha", "--learningrate", help="Set learning rate(Alpha)", type=float, default=0.01)
    parser.add_argument("-lam", help="Set lambda", type=float, default=0.1)
    parser.add_argument("-n", "--neg", help="Set sample times per epoch", type=int, default=100)
    parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=0)
    parser.add_argument("-rcmd", "--rcmd", help="Set recommend file path", default="recommend.dat")
    parser.add_argument("-topk", "--topk", help="Set recommend top k elements", type=int, default=10)
    parser.add_argument("-part", "--part", help="Split data set into k part", type=int, default=4)

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
    part = args.part
    
    main()
