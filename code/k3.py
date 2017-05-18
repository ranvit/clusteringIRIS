from __future__ import division
import sys
from collections import defaultdict
import random
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets
# from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
from itertools import count
from functools import partial

# Get iris dataset from http://archive.ics.uci.edu/ml/datasets/Iris
def load_data():
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return dict(zip(features, labels))

def dist2(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))

def dist4(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return (np.dot(d, d))

def mean(feats):
    return tuple(np.mean(feats, axis=0))

def assign(centers, features):
    # new_centers = defaultdict(list)
    # for cx in centers:
    #     for x in centers[cx]:
    #         best = min(centers, key=lambda c: dist2(x,c))
    #         new_centers[best] += [x]
    # return new_centers

    new_centers = defaultdict(list)
    for i in features:
        best = min(centers, key=lambda c: dist2(i,c))
        new_centers[best] += [i]
    return new_centers


def update(centers):
    new_centers = {}
    for c in centers:
        new_centers[mean(centers[c])] = centers[c]
    return new_centers

def kmeans(features, k, maxiter=100):
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    # print maxiter
    for i in xrange(maxiter):
        new_centers = assign(centers, features)
        new_centers = update(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
    return centers

def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)

def ss_total(clusters):
    total = 0.0
    for cx in clusters:
        for x in clusters[cx]:
            total += dist4(cx, x)
    return total

def min_extract(t):
    mins = []
    temp = 0
    for i in range(4, 38, 4):
        # print (t[temp:i])
        mins.append(min(t[temp:i]))
        temp = i
    # print mins
    return mins

def min_extract_clust(t, c):
    mins = []
    w = []
    temp = 0
    for i in range(4, 38, 4):
        # print (t[temp:i])
        f = t.index(min(t[temp:i]))
        mins.append(t[f])
        w.append(c[f])
        temp = i
    # print mins
    return (mins, w)


def demo1(seed=123):
    data = load_data()
    features = data.keys()
    random.seed(seed)
    random.shuffle(features)
    clusters = kmeans(features, 4, 100)
    for c in clusters:
        print counter([data[x] for x in clusters[c]])
    print ss_total(clusters)

def demo2(seed = 123):
    data = load_data()
    features = data.keys()
    random.seed(seed)
    maxiter_list = [5, 10, 20]
    k_list = [3, 4, 5]
    clust_list = []

    tot_list = []

    for k in k_list:
        for m in maxiter_list:
            for i in range(4):
                random.shuffle(features)
                clusters = kmeans(features, k, m)
                clust_list.append(clusters)
                total = ss_total(clusters)
                tot_list.append(total)
                # print(str(i+1) + ". k: " + str(k) + ", max iter: " + str(m) + ", SS_total: " + str(total))
            # print
        # print("-------------------------------------------------")
    return (tot_list, clust_list, data)

def demo3(t):
    print len(t)
    # plt.plot([5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20], t[:12], "ro")
    # plt.ylabel("SS_totals")
    # plt.xlabel("Max Iterations")
    # plt.title("SS_totals for k = 3")
    # plt.axis([0, 25, 70, 150])
    # plt.show()
    # plt.close()

    # plt.plot([5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20], t[12:24], "go")
    # plt.ylabel("SS_totals")
    # plt.xlabel("Max Iterations")
    # plt.title("SS_totals for k = 4")
    # plt.axis([0, 25, 50, 75])
    # plt.show()
    # plt.close()

    # plt.plot([5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20], t[24:], "bo")
    # plt.ylabel("SS_totals")
    # plt.xlabel("Max Iterations")
    # plt.title("SS_totals for k = 5")
    # plt.axis([0, 25, 45, 120])
    # plt.show()
    # plt.close()

    mins = min_extract(t)
    print mins

    plt.plot([5, 10, 20], mins[0:3], "ro")
    # plt.plot([5, 10, 20], mins[3:6], "go")
    # plt.plot([5, 10, 20], mins[6:9], "bo")
    plt.axis([0, 25, 77.5, 78.5])
    plt.ylabel("SS_totals")
    plt.xlabel("Max Iterations")
    plt.title("Minimum SS_totals for k = 3")
    plt.show()
    plt.close()

    # plt.plot([5, 10, 20], mins[0:3], "ro")
    plt.plot([5, 10, 20], mins[3:6], "go")
    # plt.plot([5, 10, 20], mins[6:9], "bo")
    plt.axis([0, 25, 56, 57])
    plt.ylabel("SS_totals")
    plt.xlabel("Max Iterations")
    plt.title("Minimum SS_totals for k = 4")
    plt.show()
    plt.close()

    # plt.plot([5, 10, 20], mins[0:3], "ro")
    # plt.plot([5, 10, 20], mins[3:6], "go")
    plt.plot([5, 10, 20], mins[6:9], "bo")
    plt.axis([0, 25, 49, 50])
    plt.ylabel("SS_totals")
    plt.xlabel("Max Iterations")
    plt.title("Minimum SS_totals for k = 5")
    plt.show()
    plt.close()

def demo4(t, c, d):
    saved_clust = min_extract_clust(t, c)
    true_count = [48, 49, 50]
    true_prop = [q/147 for q in true_count]
    # print true_prop
    # for i, j in saved_clust:
        # print
        # print i
        # for p in j:
            # print counter([d[x] for x in j[p]])
    F_l = calc(saved_clust[1], d)
    # print("Iris-setosa" + "\t" + "Iris-virginica" + "\t" + "Iris-versicolor")
    for i in range(0, 25, 3):
        # print(str(F_l[i]) + "\t\t" + str(F_l[i+1]) + "\t\t" + str(F_l[i+2]))
        avg = ((F_l[i]*true_prop[0]) + (F_l[i+1]*true_prop[1]) + (F_l[i+2]*true_prop[2]))
        # print avg
    return (saved_clust[1])[2]

def calc(saved_clust, d):
    label_list = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    true_count = [48, 49, 50]
    F_l = []
    for i in range(len(saved_clust)):
        if(i < 3):
            f1 = F1(saved_clust[i], 3, d)
            F_l += f1
        if(i > 2 and i < 6):
            f1 = F1(saved_clust[i], 4, d)
            F_l += f1
        if(i > 5):
            f1 = F1(saved_clust[i], 5, d)
            F_l += f1
    # print len(F_l)
    return F_l

def F1(c, order, d):
    ck  = c.keys()
    cv = c.values()
    ex_cl = len(ck)
    label_list = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    true_count = [48, 49, 50]
    # print type(cv)
    f_list = []
    # for p in c:
        # print counter([d[x] for x in c[p]])
    if(order == 3):
        l = [1, 0, 2]
        for i in range(3):
            a = ck[i]
            # a = d[cv[l[i]]]
            a = ([d[x] for x in cv[l[i]]])
            correct = a.count(label_list[i])
            recall = correct/true_count[i]
            prec = correct/len(a)
            f1 = (2*prec*recall)/(prec+recall)
            f_list.append(f1)

    if(order == 4):
        l = [1, 2, 3]
        for i in range(3):
            a = ck[i]
            # a = d[cv[l[i]]]
            a = ([d[x] for x in cv[l[i]]])
            correct = a.count(label_list[i])
            recall = correct/true_count[i]
            prec = correct/len(a)
            f1 = (2*prec*recall)/(prec+recall)
            f_list.append(f1)

    if(order == 5):
        l = [3, 1, 2]
        for i in range(3):
            a = ck[i]
            # a = d[cv[l[i]]]
            a = ([d[x] for x in cv[l[i]]])
            correct = a.count(label_list[i])
            recall = correct/true_count[i]
            prec = correct/len(a)
            f1 = (2*prec*recall)/(prec+recall)
            f_list.append(f1)
    return f_list


def demo5(q, d):
    centers = q.keys()
    points = q.values()
    points = [item for sublist in points for item in sublist]
    # print len(points)
    # print len(points[0])
    # print points
    # print
    SL = []
    SW = []
    PL = []
    PW = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []

    for k in centers:
        c1.append(k[0])
        c2.append(k[1])
        c3.append(k[2])
        c4.append(k[3])

    a = ([d[x] for x in points])
    # print a
    # print len(a)

    label_to_number = defaultdict(partial(next, count(1)))
    label_num = [(label_to_number[label], label) for label in a]
    z = []
    for u in label_num:
        z.append(u[0])

    # print z
    # print label_num

    for i in points:
        SL.append(i[0])
        SW.append(i[1])
        PL.append(i[2])
        PW.append(i[3])
    # print len(PW)
    # print SL
    # print len(c1)

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    # X_reduced = PCA(n_components=3).fit_transform(q)
    ax.scatter(SW, PL, PW, c=z, cmap=plt.cm.Paired)
    ax.scatter(c2, c3, c4, c = [1, 3, 2], marker = "*", s = 400, cmap=plt.cm.Paired)
    ax.set_title("Sepal Width  vs  Petal Length  vs  Petal Width")
    ax.set_xlabel("Sepal Width")
    # ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("Petal Length")
    # ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("Petal Width")
    # ax.w_zaxis.set_ticklabels([])

    plt.show()


if __name__ == "__main__":
    # demo1()
    t, c, d = demo2()
    # min_extract(t)
    # demo3(t)
    q = demo4(t, c, d)
    demo5(q, d)
    # for p in q:
    #     print (str(counter([d[x] for x in q[p]])) + "\t\t" + str(p))
