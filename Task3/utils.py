import numpy as np


def data_clean(infilepath, outfilepath):
    infile = open(infilepath, 'r', encoding='utf-8', errors='ignore')
    outfile = open(outfilepath, 'w', encoding='utf-8')
    for line in infile.readlines():
        line = line.split('  ')
        if line[0][0] != '1':
            continue
        i = 1
        while i < len(line) - 1:
            if line[i][0] == '[':  # 组合实体名
                word = line[i].split('/')[0][1:]
                i += 1
                while i < len(line) - 1 and line[i].find(']') == -1:
                    if line[i] != '':
                        word += line[i].split('/')[0]
                    i += 1
                word += line[i].split('/')[0].strip() + '/n '
            elif line[i].split('/')[1] == 'nr':  # 人名
                word = line[i].split('/')[0]
                i += 1
                if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                    word += line[i].split('/')[0] + '/n '
                else:
                    word += '/n '
                    i -= 1
            elif line[i].split('/')[1][0] == 'n':
                word = line[i].split('/')[0] + '/n '
            else:
                word = line[i].split('/')[0] + '/'
                word += line[i].split('/')[1][0].lower() + ' '
            outfile.write(word)
            i += 1
        outfile.write('\n')
    infile.close()
    outfile.close()


def load_data(file='./PeopleDaily_clean.txt'):
    print('Loading data...')
    X, y = [], []
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            words = line.strip().split()
            words, postag = [word[:-2] for word in words], [word[-1] for word in words]
            X.append((''.join(words),))
            label = np.array([words, postag, []])[:-1]
            y.append(label)
        X, y = np.concatenate(X), np.array(y)
        return X, y


def prf(p, r, c):
    P = c / p
    R = c / r
    F = 2 * P * R / (P + R)
    return P, R, F


def prf_postag(pred, gt):
    p_num, r_num, corrent_num = 0, 0, 0
    for pred_text, gt_text in zip(pred, gt):
        p_num += len(pred_text)
        r_num += len(gt_text)
        corrent_num += sum([p == g for p, g in zip(pred_text, gt_text)])
    return prf(p_num, r_num, corrent_num)


def prf_segmentation(pred, gt):
    p_num, r_num, corrent_num = 0, 0, 0
    for pred_text, gt_text in zip(pred, gt):
        p_num += len(pred_text)
        r_num += len(gt_text)
        pred_point, pred_len, gt_point, gt_len = 0, 0, 0, 0
        while pred_point < len(pred_text) and gt_point < len(gt_text):
            if pred_text[pred_point] == gt_text[gt_point]:
                corrent_num += 1
                l = len(pred_text[pred_point])
                pred_point, pred_len, gt_point, gt_len = pred_point + 1, pred_len + l, gt_point + 1, gt_len + l
            else:
                if pred_len < gt_len:
                    pred_point, pred_len = pred_point + 1, pred_len + len(pred_text[pred_point])
                else:
                    gt_point, gt_len = gt_point + 1, gt_len + len(gt_text[gt_point])
    return prf(p_num, r_num, corrent_num)


if __name__ == '__main__':
    load_data()
