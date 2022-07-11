import cupy as np
from tqdm import tqdm

ml_folder = "./ml-1m/"

def matrix_factorization(R, P, Q, K, rates_lst, steps=5000, alpha=0.0002, beta=0.02):
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
        p_minus_rate = np.count_nonzero(R < 4, axis=1) * alpha * beta
        p_minus_rate = 1 - p_minus_rate
        nP = P
        #nP = P * p_minus_rate.reshape(len(p_minus_rate), 1)
        nP = np.add(nP, (np.dot(e_table, Q.T))*(2*alpha))

        q_minus_rate = np.count_nonzero(R < 4, axis=0) * alpha * beta
        q_minus_rate = 1 - q_minus_rate
        #Q *= q_minus_rate
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
        e_table = np.subtract(R, np.dot(P, Q))
        
        e_table = np.where(R == 0, 0, e_table**2)
        e = np.sum(e_table)
        progress.set_description("Loss = {L:.2f}".format(L=e))
        
        #input()
        # 0.001: local minimum
        if e < 0.001:
            break
        
    return P, Q.T

def main():
    movie_i2id = {}
    movie_id2i = {}
    user_i2id = {}
    user_id2i = {}
    rates_lst = []

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

    R = np.zeros((N, M), dtype=np.float64)
    
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
    for i in range(len(R_lines)):
        line = R_lines[i]
        u_id, m_id, rate, timestamp = line.split("::")
        u_i = int(user_id2i[u_id])
        m_i = int(movie_id2i[m_id])
        rate = int(rate)
        R[u_i][m_i] = rate
        rates_lst.append([u_i, m_i])

    K = 100

 
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    nP, nQ = matrix_factorization(R, P, Q, K, rates_lst, alpha=0.00002, steps=1000)

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

if __name__ == '__main__':
    main()

