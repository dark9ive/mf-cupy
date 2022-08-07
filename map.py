import argparse
from tqdm import tqdm

ml_folder = ""
recommend_file = ""
k = 0

def main():
    RF = open(ml_folder + "ratings.dat", encoding="utf-8", errors='ignore')
    RF_lines = RF.readlines()
    RF.close()
    R = open(recommend_file, encoding="utf-8", errors='ignore')
    R_lines = R.readlines()
    R.close()

    user_viewed = {}
    user_recommend = {}

    for i in tqdm(range(len(RF_lines))):
        line = RF_lines[i]
        u_id, mov_id, _, _ = line.split("::")
        if u_id not in user_viewed:
            user_viewed[u_id] = [mov_id]
        else:
            user_viewed[u_id].append(mov_id)

    for i in tqdm(range(len(R_lines))):
        line = R_lines[i]
        u_id, mov_id, _ = line.split("::")
        if u_id not in user_recommend:
            user_recommend[u_id] = [mov_id]
        else:
            user_recommend[u_id].append(mov_id)

    MAP = 0
    precision_at_k = 0
    for usr in tqdm(user_recommend):
        hit = 0
        AP = 0
        for i in range(k):
            rcmd_mov = user_recommend[usr][i]
            if rcmd_mov in user_viewed[usr]:
                hit += 1
                AP += hit/(i+1)
        AP = AP / k
        MAP += AP
        precision_at_k += hit / k

    MAP = MAP / len(user_recommend)
    precision_at_k = precision_at_k / len(user_recommend)
    print("MAP @ {} = {}".format(k, MAP))
    print("Precision @ {} = {}".format(k, precision_at_k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path", help="Specify dataset path", default="./ml-1m/")
    parser.add_argument("-k", "--k", help="Set map at top k elements", type=int, default=10)
    parser.add_argument("-rcmd", "--rcmd", help="Set recommend file path", default="recommend.dat")

    args = parser.parse_args()

    ml_folder = args.path
    recommend_file = args.rcmd
    k = args.k
    
    main()
