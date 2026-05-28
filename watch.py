import pickle
import sys, os
import numpy as np
import math

pr = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])
def splitCamel(token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
p = pickle.load(open(pr + 'res_%d_%s_%s.pkl'%(seed,lr,batch_size), 'rb'))
f = pickle.load(open(pr + '.pkl', 'rb'))

print(len(f), len(p))
#assert(0)
score = []
score2 = []
eps = {}
best_ids = []
for _, i in enumerate(p):
    maxn = 1e9
    xs = p[i]
    score.extend(xs[0])
    # print(xs)
    # print(i, xs[0], xs[1], xs[3])
    # print('###############')
    # print(i)
    # print('--------------')
    # print(xs[0])
    # print('--------------')
    # print(xs[1])
    # print('--------------')
    # print(xs[3])
    # print('###############')
    minl = 1e9
    for x in f[i]['ans']:
        m = xs[1].index(x)
        minl = min(minl, m)
    score2.append(minl)
    rrdic = {}
    # for x in f[i]['methods']:
    #     rrdic[f[i]['methods'][x]] = x#".".join(x.split(":")[0].split(".")[-2:])
    #rrdict = {}
    #for s in f[i]['ftest']:
    #    rrdict[f[i]['ftest'][s]] = ".".join(s.split(":")[0].split(".")[-2:])
    # for x in f[i]['ftest']:
    #     print(splitCamel(".".join(x.split(":")[0].split(".")[-2:])), x, ".".join(x.split(":")[0].split(".")[-2:]))
    # print("-----")
    # for x in f[i]['ans']:
    #     print(splitCamel(rrdic[x]), rrdic[x], ',')
    # print("-----")
    # print(rrdic, f[i]['ans'])
    # print(splitCamel(rrdic[xs[1][0]]), rrdic[xs[1][0]], ',', xs[1][0], f[i]['ans'])
    #print(f[i]['methods'], f[i]['ftest'], f[i]['ans'])
    for x in xs[2]:
        if x in eps:
            eps[x] += 1
        else:
            eps[x] = 1
    if 10 in xs[2]:
        best_ids.append(i)
    #print(xs[2])
    #score.append(maxn)

print(score)

with open(pr + 'result_final_%d_%s_%s'%(seed,lr, batch_size), 'w') as pp:
    pp.write("lr: %f seed %d batch_size %d\n"%(lr, seed, batch_size))
    pp.write('num: %s\n'%len(p))
    # pp.write('%d: %d\n'%(10, eps[10]))
    pp.write(str(sorted(eps.items(), key=lambda x:x[1])))

# print(len(score))
a = []
for i, x in enumerate(score):
    if x != 0:
        a.append(i)
# print(a)
# print(len(score))
# print(score.count(0))
# print(score2.count(0))
# print(eps)
c1 = 0
for x in score:
    if x < 3:
        c1 += 1
c2 = 0
for x in score:
    if x < 5:
        c2 += 1
# print('top35',c1, c2)
# print(sorted(eps.items(), key=lambda x:x[1]))

# print(best_ids)
# print(len(best_ids))


best_epoch = sorted(eps.items(), key=lambda x:x[1])[-1][0]
top1 = 0
top3 = 0
top5 = 0
top10 = 0
mfr = []
mar = []
for idx in p:
    xs = p[idx]
    each_epoch_pred = xs[3]
    best_pred = each_epoch_pred[best_epoch]
    score_pred = each_epoch_pred[str(best_epoch)+'_pred']
    print('-'*20)
    print('Project Number:', idx)
    print('Correct Answer:', f[idx]['ans'])
    print(best_pred)
    print(score_pred)
    ar = []
    minl = 1e9
    to1 = 0
    to3 = 0
    to5 = 0
    to10 = 0
    for x in f[idx]['ans']:
        m = best_pred.index(x)
        ar.append(m)
        minl = min(minl, m)
    if minl == 0:
        top1 += 1
        to1 = 1
    if minl < 3:
        top3 += 1
        to3 = 1
    if minl < 5:
        top5 += 1
        to5 = 1
    if minl < 10:
        top10 += 1
        to10 = 1
    mfr.append(minl)
    mar.append(np.mean(ar))
    print('Top1:', to1)
    print('Top3:', to3)
    print('Top5:', to5)
    print('Top10:', to10)
    print('-'*20)
result_path = os.path.join("result-all")
if not os.path.exists(result_path):
    os.makedirs(result_path)

print('------------Original-----------------')
print('top1:',top1)
print('top3:',top3)
print('top5:',top5)
print('top10:',top10)
print('mfr:',np.mean(mfr))
print('mar:',np.mean(mar))
print('-----------------------------')

# with open(result_path + '/' + pr, 'w') as f:
#     f.write('top1: %d\n'%top1)
#     f.write('top3: %d\n'%top3)
#     f.write('top5: %d\n'%top5)
#     f.write('mfr: %f\n'%np.mean(mfr))
#     f.write('mar: %f\n'%np.mean(mar))
# best_epoch = sorted(eps.items(), key=lambda x:x[1])[-1][0]
top_count = [0] * 5  # list to count correct items in each position
mfr = []
mar = []
# extra accumulators
prec10_list = []
recall10_list = []
f110_list = []
auc_list = []
ca_exam_list = []
for idx in p:
    xs = p[idx]
    each_epoch_pred = xs[3]
    best_pred = each_epoch_pred[best_epoch]
    score_pred = each_epoch_pred[str(best_epoch)+'_pred']
    # print('-'*20)
    # print('Project Number:', idx)
    # print('Correct Answer:', f[idx]['ans'])
    # print('Ranking positions:',best_pred)
    # print('Scores:',score_pred)
    ar = []
    minl = 1e9
    for x in f[idx]['ans']:
        m = best_pred.index(x)
        ar.append(m)
        minl = min(minl, m)
        if m < len(top_count):  # increment the count if the index is within the list length
            top_count[m] += 1
    mfr.append(minl)
    mar.append(np.mean(ar))

    # extra metrics
    n_pos = len(f[idx]['ans'])
    n_total = len(best_pred)
    n_neg = max(n_total - n_pos, 1)

    hits10 = sum(1 for r in ar if r < 10)
    prec10 = hits10 / 10.0
    rec10 = hits10 / float(n_pos)
    f110 = 0.0 if prec10 + rec10 == 0 else 2 * prec10 * rec10 / (prec10 + rec10)
    prec10_list.append(prec10)
    recall10_list.append(rec10)
    f110_list.append(f110)

    sum_ranks = sum(ar)
    auc_val = 1.0 - (sum_ranks - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    auc_list.append(auc_val)

    ca_exam_list.append((minl + 1) / float(n_total))

    # calculate top-k values
    top1 = top_count[0]
    top3 = sum(top_count[:3])
    top5 = sum(top_count)

    # print('Current Top1:', top1)
    # print('Current Top3:', top3)
    # print('Current Top5:', top5)
    # print('-'*20)

# result_path = os.path.join("result-all")
# if not os.path.exists(result_path):
#     os.makedirs(result_path)

print('------------GBMFL-----------------')
print('Final top1:',top1)
print('Final top3:',top3)
print('Final top5:',top5)
print('mfr:',np.mean(mfr))
print('mar:',np.mean(mar))
print('-----------------------------')

# 新增评估：MRR、MAP、NDCG@1/3/5、Recall@1/3/5/10
def dcg(rank_list, k):
    dcg_val = 0.0
    for r in rank_list:
        if r < k:
            dcg_val += 1.0 / math.log2(r + 2)
    return dcg_val

def ndcg(rank_list, rel, k):
    if not rank_list:
        return 0.0
    dcg_val = dcg(rank_list, k)
    idcg_val = dcg(list(range(min(rel, k))), k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

mrr = []
map_list = []
ndcg1 = []
ndcg3 = []
ndcg5 = []
recall1 = []
recall3 = []
recall5 = []
recall10 = []

for idx in p:
    xs = p[idx]
    each_epoch_pred = xs[3]
    best_pred = each_epoch_pred[best_epoch]
    gold = f[idx]['ans']

    # ranks of all golds
    ranks = [best_pred.index(g) for g in gold]
    min_rank = min(ranks)

    # MRR
    mrr.append(1.0 / (min_rank + 1))

    # MAP
    ranks_sorted = sorted(ranks)
    ap_acc = 0.0
    for i, r in enumerate(ranks_sorted):
        ap_acc += (i + 1) / (r + 1)
    map_list.append(ap_acc / len(ranks_sorted))

    # NDCG
    ndcg1.append(ndcg(ranks, len(gold), 1))
    ndcg3.append(ndcg(ranks, len(gold), 3))
    ndcg5.append(ndcg(ranks, len(gold), 5))

    # Recall@k
    recall1.append(1 if min_rank < 1 else 0)
    recall3.append(1 if min_rank < 3 else 0)
    recall5.append(1 if min_rank < 5 else 0)
    recall10.append(1 if min_rank < 10 else 0)

print('------------Extra Metrics-----------------')
print('MRR:', np.mean(mrr))
print('MAP:', np.mean(map_list))
print('NDCG@1:', np.mean(ndcg1))
print('NDCG@3:', np.mean(ndcg3))
print('NDCG@5:', np.mean(ndcg5))
print('Recall@1:', np.mean(recall1))
print('Recall@3:', np.mean(recall3))
print('Recall@5:', np.mean(recall5))
print('Recall@10:', np.mean(recall10))
print('Precision@10:', np.mean(prec10_list))
print('F1@10:', np.mean(f110_list))
print('AUC:', np.mean(auc_list))
print('CA-EXAM:', np.mean(ca_exam_list))
print('-----------------------------')
# with open(result_path + '/' + pr, 'w') as f:
#     f.write('top1: %d\n' % top1)
#     f.write('top3: %d\n' % top3)
#     f.write('top5: %d\n' % top5)
#     f.write('mfr: %f\n' % np.mean(mfr))
#     f.write('mar: %f\n' % np.mean(mar))
