import argparse


import numpy as np
import tqdm

bert_scorer = None
metric_cache = dict()  # cache some comparison operations


def parse_table_element_to_relation(table, i, j, row_header: bool, col_header: bool):
    assert row_header or col_header
    relation = []
    if row_header:
        assert j > 0
        relation.append(table[i][0])
    if col_header:
        assert i > 0
        relation.append(table[0][j])
    relation.append(table[i][j])
    return tuple(relation)


def parse_table_to_data(table, row_header: bool, col_header: bool):  # ret: row_headers, col_headers, relation tuples
    if is_empty_table(table, row_header, col_header):
        return set(), set(), set()

    assert row_header or col_header
    row_headers = list(table[:, 0]) if row_header else []
    col_headers = list(table[0, :]) if col_header else []
    if row_header and col_headers:
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    row, col = table.shape
    relations = []
    for i in range(1 if col_header else 0, row):
        for j in range(1 if row_header else 0, col):
            if table[i][j] != "":
                relations.append(parse_table_element_to_relation(table, i, j, row_header, col_header))
    return set(row_headers), set(col_headers), set(relations)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('tgt')
    parser.add_argument('--row-header', default=False, action="store_true")
    parser.add_argument('--col-header', default=False, action="store_true")
    parser.add_argument('--table-name', default=None)
    parser.add_argument('--metric', default='E', choices=['E', 'c', 'BS-scaled', ],
                        help="E: exact match\nc: chrf\nBS-scaled: re-scaled BERTScore")
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


def calc_similarity_matrix(tgt_data, pred_data, metric):
    # print(tgt_data, pred_data)
    # print("================")
    def calc_data_similarity(tgt, pred):
        if isinstance(tgt, tuple):
            ret = 1.0
            for tt, pp in zip(tgt, pred):
                ret *= calc_data_similarity(tt, pp)
            return ret

        if (tgt, pred) in metric_cache:
            return metric_cache[(tgt, pred)]

        if metric == 'E':
            ret = int(tgt == pred)

        metric_cache[(tgt, pred)] = ret
        return ret

    return np.array([[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data], dtype=np.float)


def metrics_by_sim(tgt_data, pred_data, metric):
    sim = calc_similarity_matrix(tgt_data, pred_data, metric)  # (n_tgt, n_pred) matrix
    prec = np.mean(np.max(sim, axis=0))
    recall = np.mean(np.max(sim, axis=1))
    if prec + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def extract_header_by_name(text, name):
    col_list = ['Wins', 'Losses', 'Total points', 'Percentage of field goals', 'Number of team assists', 'Percentage of 3 points', 'Turnovers', 'Rebounds', 'Points in 1st quarter', 'Points in 2nd quarter', 'Points in 3rd quarter', 'Points in 4th quarter']

    rows = []
    if name == "Team":
        # text = text.split('<columns2> ')[0]
        try:
            columns = text.split('>')
            tmp = []
            for i in columns:
                if "TRUE" in i:
                    i = i.split(" <")[0]
                    if i[0] == " ":
                        i = i[1:]
                    if i in col_list:
                        tmp.append(i)
            columns = tmp
        except:
            print(text)
            return ([], [])
        
        # print(columns, rows)

    # if (columns, rows) == ([], []):
    #     print(a)
    #     print("asdas")
    return (columns, rows)

def is_empty_table(headers):
    col = headers[0]
    if col == []:
        return True
    else:
        return False
    
def devide_by_len(pred_lns, tgt_lns):
    import re
    max_card = 0
    for tgt in tgt_lns:
        if max_card < len(re.split("[,.?]\s*", tgt)):
            max_card = len(re.split("[,.?]\s*", tgt))
    pred_bucket = {}
    tgt_bucket = {}
    for i in range(1, max_card + 1):
        pred_bucket[i] = []
        tgt_bucket[i] = []
    import re
    for true_labels, predicted_labels in zip(tgt_lns, pred_lns):
        tgt_bucket[len(re.split("[,.?]\s*", true_labels))].append(true_labels)
        pred_bucket[len(re.split("[,.?]\s*", true_labels))].append(predicted_labels)

    return pred_bucket, tgt_bucket


def get_extract_metrics(pred_lns, tgt_lns, label_constraint, decoding_format='tree'):
    import re
    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)

    def micro_metrics(pred_lns, tgt_lns):
        p = 0.
        r = 0.
        pred_label_count = 0.
        pred_label_hit = 0.
        gold_label_count = 0.

        for true_labels, predicted_labels in zip(tgt_lns, pred_lns):
            true_labels = list(set(re.split("[,.?]\s*", true_labels)))
            predicted_labels = list(set(re.split("[,.?]\s*", predicted_labels)))

            if predicted_labels:
                pred_label_count += len(predicted_labels)
                pred_label_hit += len(set(predicted_labels).intersection(set(true_labels)))

            if len(true_labels):
                gold_label_count += len(true_labels)

        if pred_label_count > 0:
            precision = pred_label_hit / pred_label_count
        if gold_label_count > 0:
            recall = pred_label_hit / gold_label_count
        return precision, recall, f1(precision, recall)


    def macro_metrics(pred_lns, tgt_lns):
        types = {}
        for tgt in tgt_lns:
            true_labels = list(set(re.split("[,.?]\s*", tgt)))
            for label in true_labels:
                if label not in types.keys():
                    types[label] = [0., 0., 0.] # h, p, g

        for true_labels, predicted_labels in zip(tgt_lns, pred_lns):
            
            true_labels = list(set(re.split("[,.?]\s*", true_labels)))
            predicted_labels = list(set(re.split("[,.?]\s*", predicted_labels)))
            print(predicted_labels)
            for predicted_label in predicted_labels:
                if predicted_label in types.keys():
                    types[predicted_label][1] += 1
                if predicted_label in true_labels:
                    types[predicted_label][0] += 1

            for true_label in true_labels:
                types[true_label][2] += 1

        t_precision = 0.
        t_recall = 0.
        f1 = 0.
        for h, p, g in types.values():
            if p != 0:
                precision = h / p
            else:
                precision = 0.
            if g != 0:
                recall = h / g
            else:
                recall = 0.
            t_precision += precision
            t_recall += recall

            if precision != 0. and recall != 0.:
                f1 += 2 * (precision * recall) / (precision + recall)
        if len(types.items()) == 0:
            types['none'] = 1
        return t_precision / len(types.items()), t_recall / len(types.items()), f1 / len(types.items())

    def sample_metrics(pred_lns, tgt_lns):
        num_examples = len(pred_lns)
        p = 0.
        r = 0.
        pred_example_count = 0.
        pred_label_count = 0.
        gold_label_count = 0.

        for true_labels, predicted_labels in zip(tgt_lns, pred_lns):

            true_labels = list(set(re.split("[,.?]\s*", true_labels)))
            predicted_labels = list(set(re.split("[,.?]\s*", predicted_labels)))

            if predicted_labels:
                pred_example_count += 1
                pred_label_count += len(predicted_labels)
                per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                p += per_p
            if len(true_labels):
                gold_label_count += 1
                per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
                r += per_r
        if pred_example_count > 0:
            precision = p / pred_example_count
        else:
            precision = 0
        if gold_label_count > 0:
            recall = r / gold_label_count
        else:
            recall = 0

        return precision, recall, f1(precision, recall)
    
    def precision1(pred_lns, tgt_lns):
        hit = 0
        total = 0
        for true_labels, predicted_labels in zip(tgt_lns, pred_lns):
            true_labels = re.split("[,.?]\s*", true_labels)
            predicted_label = re.split("[,.?]\s*", predicted_labels)[0]

            if predicted_label != "":
                if predicted_label in true_labels:
                    hit += 1
                    total += 1
                else:
                    total += 1
        if total == 0:
            prec_1 = 0
        else:
            prec_1 = hit / total
        return prec_1
    def class_metrics(pred_lns, tgt_lns):
        types = {}
        class_f1 = {}
        for tgt in tgt_lns:
            true_labels = list(set(re.split("[\,\.\?]*\s", tgt)))
            for label in true_labels:
                if label not in types.keys():
                    types[label] = [0., 0., 0.] # h, p, g
                    class_f1[label] = 0.# h, p, g

        for true_labels, predicted_labels in zip(tgt_lns, pred_lns):
            true_labels = list(set(re.split("[\,\.\?]*\s", true_labels)))
            predicted_labels = list(set(re.split("[\,\.\?]*\s", predicted_labels)))

            for predicted_label in predicted_labels:
                if predicted_label in types.keys():
                    types[predicted_label][1] += 1
                if predicted_label in true_labels:
                    types[predicted_label][0] += 1

            for true_label in true_labels:
                types[true_label][2] += 1
        
        for key, (h, p, g) in types.items():
            if p != 0:
                precision = h / p
            else:
                precision = 0.
            if g != 0:
                recall = h / g
            else:
                recall = 0.

            if precision != 0. and recall != 0.:
                class_f1[key] = 2 * (precision * recall) / (precision + recall)
        return class_f1
    
    micro_p, micro_r, micro_f1 = micro_metrics(pred_lns, tgt_lns)
    macro_p, macro_r, macro_f1 = macro_metrics(pred_lns, tgt_lns)
    sample_p, sample_r, sample_f1 = sample_metrics(pred_lns, tgt_lns)
    prec_1 = precision1(pred_lns, tgt_lns)
    class_dic = class_metrics(pred_lns, tgt_lns)
    pred_bucket, tgt_bucket = devide_by_len(pred_lns, tgt_lns)
    devide_macro_ps, devide_macro_rs, devide_macro_f1s = [], [], []
    for pred_lns, tgt_lns in zip(pred_bucket.values(), tgt_bucket.values()):
        devide_macro_p, devide_macro_r, devide_macro_f1 = macro_metrics(pred_lns, tgt_lns)
        devide_macro_ps.append(devide_macro_p)
        devide_macro_rs.append(devide_macro_r)
        devide_macro_f1s.append(devide_macro_f1)
    # print(devide_macro_ps, devide_macro_rs, devide_macro_f1s)
    result = {"sample_p": sample_p, "sample_r": sample_r, "sample_f1": sample_f1, "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1, "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1, "precision@1": prec_1}
    # result['ship_f1'] = class_dic['ship']
    # result['veg-oil_f1'] = class_dic['veg-oil']
    # result['cpi_f1'] = class_dic['cpi']
    return result

