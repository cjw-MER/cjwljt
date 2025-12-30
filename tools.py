import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model
import pickle
import numpy as np
from utils import *

def retrieval_topk_load(model_file, user_id="956"):
    top_10_users = load_json_file(model_file)
    return [top_10_users[user_id]]

def stdout_retrived_items_load(item_id, score=None, item_name=None):
    retrived_items = []
    item_database = load_json_file('/home/liujuntao/Agent4Rec/data/ml-1m/ml-1m.item.json')
    for n in range(len(item_id)):
        item_strings = ""
        for iid in item_id[n]:
            item_str = f"item_{iid}"
            # item_strings = item_strings + str(iid) + ', ' + str(ina) + ", " + str(round(s.item(), 4)) + "\n"
            item_strings = item_strings + f"""item_ID:item_{iid}, title:{item_database[item_str]["movie_title"]}, release_year:{item_database[item_str]["release_year"]}, genre:{item_database[item_str]["genre"]}""" + "\n"
        retrived_items.append(item_strings)
    return retrived_items

def retrieval_topk(model_file, user_id=None, topK=10):  
    # load trained model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file,
    )

    # retrieval top K items, and the corresponding score.
    
    uid_series = dataset.token2id(dataset.uid_field, user_id)
    topk_score, topk_iid_list = full_sort_topk(
        uid_series, model, test_data, k=topK, device=config["device"]
    )
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    external_item_list_name = []
    for u_list in external_item_list:
        external_item_list_name.append(u_list)
    external_item_list_name = np.array(external_item_list_name)

    return topk_score, external_item_list, external_item_list_name

def stdout_retrived_items(item_id, score=None, item_name=None):
    retrived_items = []
    item_database = load_json_file('data/ml-1m/ml-1m.item.json')
    for n in range(item_id.shape[0]):
        item_strings = ""
        for iid in item_id[n]:
            item_str = f"item_{iid}"
            # item_strings = item_strings + str(iid) + ', ' + str(ina) + ", " + str(round(s.item(), 4)) + "\n"
            item_strings = item_strings + f"""item_ID:item_{iid}, title:{item_database[item_str]["movie_title"]}, release_year:{item_database[item_str]["release_year"]}, genre:{item_database[item_str]["genre"]}""" + "\n"
        retrived_items.append(item_strings)
    return retrived_items


if __name__ == "__main__":
    
#     # test

#     # score = full_sort_scores(uid_series, model, test_data, device=config["device"])
#     # print(score)  # score of all items
#     # print(
#     #     score[0, dataset.token2id(dataset.iid_field, ["242", "302"])]
#     # )  # score of item ['242', '302'] for user '196'.
    users = ["956"]
    topK = 10
    topk_score, external_item_list, external_item_list_name = retrieval_topk(model_file="/home/chengjiawei/Agent4Rec/saved/SASRec_ml-1m_50.pth", user_id=users, topK=topK)
    retrived_items = stdout_retrived_items(external_item_list)
    print(retrived_items)