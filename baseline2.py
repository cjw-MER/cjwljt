import numpy as np
from collections import defaultdict
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import json
from metric import ndcg_all, recall_all
import random

def metric_cal(candidate_items=None):

    with open("/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m_user_inter50.json", "r", encoding="utf-8") as f:
        selected_users = json.load(f)

    user_id = list(selected_users.keys())

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="/home/liujuntao/RecBole-master/saved_50/SASRec-Dec-25-2025_20-30-57.pth",
    )

    # 随机选取真实item的token来作为候选列表
    token_items = dataset.field2token_id['item_id']   # dict: token -> inner_id
    token_users = dataset.field2token_id['user_id']
    all_itemtoken = list(token_items.keys())

    uid_series = dataset.token2id(dataset.uid_field, user_id)

    if candidate_items == None: # 用于计算训练好的推荐模型的性能
        topk_score, topk_iid_list = full_sort_topk(
            uid_series, model, test_data, k=10, device=config["device"]
        )
        target = {}
        uid_target = {}
        random_50 = {}
        data = test_data.dataset.inter_feat.interaction
        for (uid, iid, iid_his, i_len, iid_hisScore) in zip(data['user_id'], data['item_id'], data['item_id_list'], data['item_length'], data['rating_list']):
            for ids in uid_series:
                if ids == uid:
                    # 将每个user的target存到一个列表中
                    uid_token = dataset.id2token('user_id', ids)
                    real_token = dataset.id2token('item_id', iid_his[i_len-1])
                    target[uid_token] = real_token

                    # 下面是计算指标用的
                    uid_target[ids]=iid_his[i_len-1].detach().cpu()
        # for user in user_id:
        #     user_target = target[user]
        #     pool = [x for x in all_itemtoken if x != user_target]
        #     new_list = random.sample(pool, 199)
        #     pos = random.randint(0, len(new_list))
        #     new_list.insert(pos, user_target)
        #     random_50[user] = new_list
        # with open("/home/chengjiawei/Agent4Rec/data/ml-1m/test_topk_prediction_random_200.json", "w", encoding="utf-8") as f:
        #     json.dump(random_50, f, ensure_ascii=False, indent=2)
        # exit()
        vals = [int(uid_target[k]) for k in uid_series]
        print(recall_all(topk_iid_list, vals, 3))
        print(recall_all(topk_iid_list, vals, 5))
        print(ndcg_all(topk_iid_list, vals, 3))
        print(ndcg_all(topk_iid_list, vals, 5))
    else: # 用于测试我们框架生成的候选列表的性能
        topk_iid_list = test_data.dataset.token2id(dataset.iid_field, candidate_items)
        uid_target = {}
        data = test_data.dataset.inter_feat.interaction
        try:
            for (uid, iid, iid_his, i_len, iid_hisScore) in zip(data['user_id'], data['item_id'], data['item_id_list'], data['item_length'], data['rating_list']):
                for ids in uid_series:
                    if ids == uid:
                        uid_target[ids]=iid_his[i_len-1].detach().cpu()

            vals = [int(uid_target[k]) for k in uid_series]
            print("recall_3", recall_all(topk_iid_list, vals, 3))
            print("recall_5", recall_all(topk_iid_list, vals, 5))
            print("ndcg_3", ndcg_all(topk_iid_list, vals, 3))
            print("ndcg_5", ndcg_all(topk_iid_list, vals, 5))
        except:
            for (uid, iid) in zip(data['user_id'], data['item_id']):
                for ids in uid_series:
                    if ids == uid:
                        uid_target[ids]=iid_his[i_len-1].detach().cpu()

            vals = [int(uid_target[k]) for k in uid_series]
            print("recall_3", recall_all(topk_iid_list, vals, 3))
            print("recall_5", recall_all(topk_iid_list, vals, 5))
            print("ndcg_3", ndcg_all(topk_iid_list, vals, 3))
            print("ndcg_5", ndcg_all(topk_iid_list, vals, 5))
