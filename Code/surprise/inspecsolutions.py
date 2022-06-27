from surprise import accuracy
from surprise import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import statistics
import datetime

pd.options.mode.chained_assignment = None  # default='warn'


path_output_biclust = "D:/usbcf-paper/NewResults/Biclust/ml-100k/"
path_output_svd = "D:/usbcf-paper/NewResults/SVD/jester/"
path_output_useritembased = "D:/usbcf-paper/NewResults/UserItemBased/jester/"


#stats_test = pd.read_csv(path_output_biclust+"stats_test.csv")
# stats_test["stats_bic2"] = stats_test["stats_bic2"].apply(
#     lambda x: eval(x) if x != " " else x)
# stats_test["avg_row_size"] = stats_test["stats_bic2"].apply(
#     lambda x: statistics.mean(x) if x != " " else None)
# stats_test["std_row_size"] = stats_test["stats_bic2"].apply(
#     lambda x: statistics.stdev(x) if x != " " else None)

# stats_test["stats_bic3"] = stats_test["stats_bic3"].apply(
#     lambda x: eval(x) if x != " " else x)
# stats_test["avg_col_size"] = stats_test["stats_bic3"].apply(
#     lambda x: statistics.mean(x) if x != " " else None)
# stats_test["std_col_size"] = stats_test["stats_bic3"].apply(
#     lambda x: statistics.stdev(x) if x != " " else None)

# stats_test.groupby(["model"])[["avg_row_size", "avg_col_size"]
#                               ].mean().to_csv(path_output_biclust+"avg_sizes.csv")
# stats_test.groupby(["model"])[["std_row_size", "std_col_size"]
#                               ].mean().to_csv(path_output_biclust+"std_sizes.csv")


def load_results_todf(path):
    acc_results_df = pd.read_csv(path+"acc_results.csv",
                                 header=None)
    speed_results_df = pd.read_csv(path+"stats_test.csv")
    speed_results_df["avg_rows"] = speed_results_df["stats_bic2"].apply(
        lambda x:  statistics.mean(eval(x)) if x != " " else None)
    speed_results_df["avg_cols"] = speed_results_df["stats_bic3"].apply(
        lambda x:  statistics.mean(eval(x)) if x != " " else None)
    return acc_results_df, speed_results_df


acc_results_df, speed_results_df = load_results_todf(path_output_svd)
acc_results_df.columns = ["model", "fold", "coverage", "cov_user",
                          "cov_item", "mae", "rmse"]

biclust_mean_acc_results = acc_results_df.groupby(
    ["model"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].mean().reset_index()
biclust_std_acc_results = acc_results_df.groupby(
    ["model"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].std().reset_index()
mean_speed_results_df = speed_results_df.groupby(
    ["model"])[["fit_time", "pred_time", "avg_rows", "avg_cols"]].mean().reset_index()


bbcf_mean_acc_results = biclust_mean_acc_results[biclust_mean_acc_results["model"].str.startswith(
    "BBCF")]
bbcf_mean_acc_results["k"] = bbcf_mean_acc_results["model"].str.extract(
    r'(BBCF.*?\(\d+,\d+)')
bbcf_mean_acc_results["k"] = bbcf_mean_acc_results["k"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_mean_acc_results["nnbrs"] = bbcf_mean_acc_results["model"].str.extract(
    r'(BBCF.*?\(\d+)')
bbcf_mean_acc_results["nnbrs"] = bbcf_mean_acc_results["nnbrs"].apply(
    lambda x: int(x.split("(")[-1]))
bbcf_mean_acc_results["mincols"] = bbcf_mean_acc_results["model"].str.extract(
    r'(QUBIC2\(\w+,\d+,\d+)')
bbcf_mean_acc_results["mincols"] = bbcf_mean_acc_results["mincols"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_mean_acc_results["model"] = "BBCF"

bbcf_std_acc_results = biclust_std_acc_results[biclust_std_acc_results["model"].str.startswith(
    "BBCF")]
bbcf_std_acc_results["k"] = bbcf_std_acc_results["model"].str.extract(
    r'(BBCF.*?\(\d+,\d+)')
bbcf_std_acc_results["k"] = bbcf_std_acc_results["k"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_std_acc_results["nnbrs"] = bbcf_std_acc_results["model"].str.extract(
    r'(BBCF.*?\(\d+)')
bbcf_std_acc_results["nnbrs"] = bbcf_std_acc_results["nnbrs"].apply(
    lambda x: int(x.split("(")[-1]))
bbcf_std_acc_results["mincols"] = bbcf_std_acc_results["model"].str.extract(
    r'(QUBIC2\(\w+,\d+,\d+)')
bbcf_std_acc_results["mincols"] = bbcf_std_acc_results["mincols"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_std_acc_results["model"] = "BBCF"

bbcf_mean_speed_results = mean_speed_results_df[mean_speed_results_df["model"].str.startswith(
    "BBCF")]
bbcf_mean_speed_results["k"] = bbcf_mean_speed_results["model"].str.extract(
    r'(BBCF.*?\(\d+,\d+)')
bbcf_mean_speed_results["k"] = bbcf_mean_speed_results["k"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_mean_speed_results["nnbrs"] = bbcf_mean_speed_results["model"].str.extract(
    r'(BBCF.*?\(\d+)')
bbcf_mean_speed_results["nnbrs"] = bbcf_mean_speed_results["nnbrs"].apply(
    lambda x: int(x.split("(")[-1]))
bbcf_mean_speed_results["mincols"] = bbcf_mean_speed_results["model"].str.extract(
    r'(QUBIC2\(\w+,\d+,\d+)')
bbcf_mean_speed_results["mincols"] = bbcf_mean_speed_results["mincols"].apply(
    lambda x: int(x.split(",")[-1]))
bbcf_mean_speed_results["model"] = "BBCF"


usbcf_mean_acc_results = biclust_mean_acc_results[biclust_mean_acc_results["model"].str.startswith(
    "USBCF")]
usbcf_mean_acc_results["minsim"] = usbcf_mean_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d*\.?\d*)')
usbcf_mean_acc_results["minsim"] = usbcf_mean_acc_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcf_mean_acc_results["k"] = usbcf_mean_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcf_mean_acc_results["k"] = usbcf_mean_acc_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcf_mean_acc_results["mincols"] = usbcf_mean_acc_results["model"].str.extract(
    r'(\},\d+|\}.*\])')
usbcf_mean_acc_results["mincols"] = usbcf_mean_acc_results["mincols"].apply(
    lambda x: x.split("},")[-1])
usbcf_mean_acc_results["model"] = "USBCF"

usbcf_std_acc_results = biclust_std_acc_results[biclust_std_acc_results["model"].str.startswith(
    "USBCF")]
usbcf_std_acc_results["minsim"] = usbcf_std_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d*\.?\d*)')
usbcf_std_acc_results["minsim"] = usbcf_std_acc_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcf_std_acc_results["k"] = usbcf_std_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcf_std_acc_results["k"] = usbcf_std_acc_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcf_std_acc_results["mincols"] = usbcf_std_acc_results["model"].str.extract(
    r'(\},\d+|\}.*\])')
usbcf_std_acc_results["mincols"] = usbcf_std_acc_results["mincols"].apply(
    lambda x: x.split("},")[-1])
usbcf_std_acc_results["model"] = "USBCF"

usbcf_mean_speed_results = mean_speed_results_df[mean_speed_results_df["model"].str.startswith(
    "USBCF")]
usbcf_mean_speed_results["minsim"] = usbcf_mean_speed_results["model"].str.extract(
    r'(USBCF.*?\(\d*\.?\d*)')
usbcf_mean_speed_results["minsim"] = usbcf_mean_speed_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcf_mean_speed_results["k"] = usbcf_mean_speed_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcf_mean_speed_results["k"] = usbcf_mean_speed_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcf_mean_speed_results["mincols"] = usbcf_mean_speed_results["model"].str.extract(
    r'(\},\d+|\}.*\])')
usbcf_mean_speed_results["mincols"] = usbcf_mean_speed_results["mincols"].apply(
    lambda x: x.split("},")[-1])
usbcf_mean_speed_results["model"] = "USBCF"


usbcf_mean_acc_resultsminsim02_k40 = usbcf_mean_acc_results[
    usbcf_mean_acc_results["minsim"] == 0.2]
usbcf_mean_acc_resultsminsim02_k40 = usbcf_mean_acc_resultsminsim02_k40[
    usbcf_mean_acc_resultsminsim02_k40["k"] == 40]

usbcf_mean_acc_resultsminsim02_k20 = usbcf_mean_acc_results[
    usbcf_mean_acc_results["minsim"] == 0.2]
usbcf_mean_acc_resultsminsim02_k20 = usbcf_mean_acc_resultsminsim02_k20[
    usbcf_mean_acc_resultsminsim02_k20["k"] == 20]


# fig1 = plt.figure(1)
# plt.xlabel('mincols')
# plt.ylabel('Root Mean Square Error (RMSE)')
# plt.plot(usbcf_mean_acc_resultsminsim02.mincols,usbcf_mean_acc_resultsminsim02.rmse ,marker='x', markersize=4, color = "b")
# fig1.savefig("D:/usbcf-paper/Images/Results/usbcf_minsim_rmse.pdf",bbox_inches='tight', dpi=1200)


usbcfcomb_mean_acc_results = biclust_mean_acc_results[biclust_mean_acc_results["model"].str.startswith(
    "USBCFComb")]
usbcfcomb_mean_acc_results["minsim"] = usbcfcomb_mean_acc_results["model"].str.extract(
    r'(USBCFComb\w*\(\d*\.?\d*)')
usbcfcomb_mean_acc_results["minsim"] = usbcfcomb_mean_acc_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcfcomb_mean_acc_results["k"] = usbcfcomb_mean_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcfcomb_mean_acc_results["k"] = usbcfcomb_mean_acc_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcfcomb_mean_acc_results["mincols"] = usbcfcomb_mean_acc_results["model"].str.extract(
    r'(\[.*\])')
usbcfcomb_mean_acc_results["model"] = "USBCFCombineBicSols"

usbcfcomb_std_acc_results = biclust_std_acc_results[biclust_std_acc_results["model"].str.startswith(
    "USBCFComb")]
usbcfcomb_std_acc_results["minsim"] = usbcfcomb_std_acc_results["model"].str.extract(
    r'(USBCFComb\w*\(\d*\.?\d*)')
usbcfcomb_std_acc_results["minsim"] = usbcfcomb_std_acc_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcfcomb_std_acc_results["k"] = usbcfcomb_std_acc_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcfcomb_std_acc_results["k"] = usbcfcomb_std_acc_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcfcomb_std_acc_results["mincols"] = usbcfcomb_std_acc_results["model"].str.extract(
    r'(\[.*\])')
usbcfcomb_std_acc_results["model"] = "USBCFCombineBicSols"


usbcfcomb_mean_speed_results = mean_speed_results_df[mean_speed_results_df["model"].str.startswith(
    "USBCFComb")]
usbcfcomb_mean_speed_results["minsim"] = usbcfcomb_mean_speed_results["model"].str.extract(
    r'(USBCFComb\w*\(\d*\.?\d*)')
usbcfcomb_mean_speed_results["minsim"] = usbcfcomb_mean_speed_results["minsim"].apply(
    lambda x: float(x.split("(")[-1]))
usbcfcomb_mean_speed_results["k"] = usbcfcomb_mean_speed_results["model"].str.extract(
    r'(USBCF.*?\(\d+.\d+,\d+)')
usbcfcomb_mean_speed_results["k"] = usbcfcomb_mean_speed_results["k"].apply(
    lambda x: float(x.split(",")[-1]))
usbcfcomb_mean_speed_results["mincols"] = usbcfcomb_mean_speed_results["model"].str.extract(
    r'(\[.*\])')
usbcfcomb_mean_speed_results["model"] = "USBCFCombineBicSols"


bbcf_mean_acc_results.groupby(
    ["mincols"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].mean()
bbcf_mean_acc_results[bbcf_mean_acc_results["nnbrs"] == 150][[
    "mincols", "coverage", "mae", "rmse"]].sort_values(by="mincols")

bbcf_mean_acc_results_k40 = bbcf_mean_acc_results[bbcf_mean_acc_results["k"] == 40]
bbcf_mean_speed_results_k40 = bbcf_mean_speed_results[bbcf_mean_speed_results["k"] == 40]

bbcf_mean_acc_results_k40mincols15 = bbcf_mean_acc_results_k40[bbcf_mean_acc_results_k40["mincols"] == 15][[
    "nnbrs", "coverage", "mae", "rmse"]].sort_values(by="nnbrs")
bbcf_mean_speed_results_k40mincols15 = bbcf_mean_speed_results_k40[bbcf_mean_speed_results_k40["mincols"] == 15][[
    "nnbrs", "avg_rows", "avg_cols"]].sort_values(by="nnbrs")

fig1 = plt.figure(1)
plt.xlabel('NNBics')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.plot(bbcf_mean_acc_results_k40mincols15.nnbrs,
         bbcf_mean_acc_results_k40mincols15.rmse, marker='x', markersize=4, color="b")
fig1.savefig("D:/usbcf-paper/Images/Results/bbcf_nnbics_rmse.pdf",
             bbox_inches='tight', dpi=1200)
fig2 = plt.figure(2)
plt.xlabel('NNBics')
plt.ylabel('Coverage (%)')
plt.plot(bbcf_mean_acc_results_k40mincols15.nnbrs,
         bbcf_mean_acc_results_k40mincols15.coverage*100, marker='x', markersize=4, color="b")
fig2.savefig("D:/usbcf-paper/Images/Results/bbcf_nnbics_cov.pdf",
             bbox_inches='tight', dpi=1200)


fig3 = plt.figure(3)
plt.xlabel('NNBics')
plt.ylabel('Average number of users \n per neighborhood')
plt.plot(bbcf_mean_speed_results_k40mincols15.nnbrs,
         bbcf_mean_speed_results_k40mincols15.avg_rows, marker='x', markersize=4, color="b")
fig3.savefig("D:/usbcf-paper/Images/Results/bbcf_nnbics_avgrows.pdf",
             bbox_inches='tight', dpi=1200)

fig4 = plt.figure(4)
plt.xlabel('User-Bicluster Similarity Threshold')
plt.ylabel('NNBics')
plt.plot(bbcf_mean_speed_results_k40mincols15.nnbrs,
         bbcf_mean_speed_results_k40mincols15.avg_cols, marker='x', markersize=4, color="b")
fig4.savefig("D:/usbcf-paper/Images/Results/bbcf_nnbics_avgcols.pdf",
             bbox_inches='tight', dpi=1200)

# # # usbcf_groupedminsim = usbcf_mean_acc_results.groupby(
# # #     ["minsim"])[["minsim", "coverage", "mae", "rmse"]].mean()
# # usbcf_mean_acc_results_mincols10 = usbcf_mean_acc_results[usbcf_mean_acc_results["mincols"] == 10][[
# #     "minsim", "coverage", "mae", "rmse"]].sort_values(by="minsim")

# # fig3 = plt.figure(3)
# # plt.xlabel('User-Bicluster Similarity Threshold')
# # plt.ylabel('Root Mean Square Error (RMSE)')
# # plt.plot(usbcf_mean_acc_results_mincols10.minsim,usbcf_mean_acc_results_mincols10.rmse ,marker='x', markersize=4, color = "b")
# # fig3.savefig("D:/usbcf-paper/Images/Results/usbcf_minsim_rmse.pdf",bbox_inches='tight', dpi=1200)

# # fig4 = plt.figure(4)
# # plt.xlabel('User-Bicluster Similarity Threshold')
# # plt.ylabel('Coverage (%)')
# # plt.plot(usbcf_mean_acc_results_mincols10.minsim,usbcf_mean_acc_results_mincols10.coverage*100 ,marker='x', markersize=4, color = "b")
# # fig4.savefig("D:/usbcf-paper/Images/Results/usbcf_minsim_cov.pdf",bbox_inches='tight', dpi=1200)


fig5 = plt.figure(5)
plt.xlabel('User-Bicluster Similarity Threshold')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.plot(usbcfcomb_mean_acc_results.minsim,
         usbcfcomb_mean_acc_results.rmse, marker='x', markersize=4, color="b")
fig5.savefig("D:/usbcf-paper/NewImages/Results/usbcfcomb_minsim_rmse.pdf",
             bbox_inches='tight', dpi=1200)

fig6 = plt.figure(6)
plt.xlabel('User-Bicluster Similarity Threshold')
plt.ylabel('Coverage (%)')
plt.plot(usbcfcomb_mean_acc_results.minsim,
         usbcfcomb_mean_acc_results.coverage*100, marker='x', markersize=4, color="b")
fig6.savefig("D:/usbcf-paper/NewImages/Results/usbcfcomb_minsim_cov.pdf",
             bbox_inches='tight', dpi=1200)


fig7 = plt.figure(7)
plt.xlabel('User-Bicluster Similarity Threshold')
plt.ylabel('Average number of users \n per neighborhood')
plt.plot(usbcfcomb_mean_speed_results.minsim,
         usbcfcomb_mean_speed_results.avg_rows, marker='x', markersize=4, color="b")
fig7.savefig("D:/usbcf-paper/Images/Results/usbcfcomb_minsim_avgrows.pdf",
             bbox_inches='tight', dpi=1200)

fig8 = plt.figure(8)
plt.xlabel('User-Bicluster Similarity Threshold')
plt.ylabel('Average number of items \n per neighborhood')
plt.plot(usbcfcomb_mean_speed_results.minsim,
         usbcfcomb_mean_speed_results.avg_cols, marker='x', markersize=4, color="b")
fig8.savefig("D:/usbcf-paper/Images/Results/usbcfcomb_minsim_avgcols.pdf",
             bbox_inches='tight', dpi=1200)


acc_results_df, speed_results_df = load_results_todf(path_output_biclust)
acc_results_df.columns = ["model", "fold", "coverage", "cov_user",
                          "cov_item", "mae", "rmse"]


results_complemented_df = pd.DataFrame(columns=["model", "minsim", "fold",
                                                "rmse", "mae", "coverage",
                                                "cov_user", "cov_item"])

for simthreshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for k in [40]:
        for fold in range(1, 6):
            print("Fold="+str(fold))
            preds_usbcf, _ = dump.load(path_output_biclust + "iteration" +
                                       str(fold) + "/"+"USBCFCombineBicSols_nomem("
                                       + str(simthreshold) + "," + str(k)
                                       + ",KNNMeans,{'name'_ 'pearson', 'user_based'_ False, 'min_support'_ 1},[3, 5, 7, 10, 15, 20]).bpk")
            all_test_users = set([pred[0] for pred in preds_usbcf])
            all_test_items = set([pred[1] for pred in preds_usbcf])
    
            real_preds_usbcf = list(
                filter(lambda x: not x[4]["was_impossible"], preds_usbcf))
            impossible_usbcf_preds = list(
                filter(lambda x:  x[4]["was_impossible"], preds_usbcf))
            
            #free memory
            preds_usbcf_size = len(preds_usbcf)
            preds_usbcf = None
            
            preds_coclust, _ = dump.load(
                path_output_svd + "iteration" + str(fold) + "/"+"Coclust.bpk")
            usbcf_complemented = real_preds_usbcf.copy()
            coclus_impossible = []
            for pred_coclust in preds_coclust:
                impossible_flag = False
                for pred_usbcf in impossible_usbcf_preds:
                    if pred_coclust[0] == pred_usbcf[0] and pred_coclust[1] == pred_usbcf[1]:
                        impossible_flag = True
                        break
                if impossible_flag:
                    coclus_impossible.append(pred_coclust)
                    
            #free memory
            preds_coclust = None
            
            usbcf_complemented.extend(coclus_impossible)
            users_considered = set([pred[0] for pred in usbcf_complemented])
            items_considered = set([pred[1] for pred in usbcf_complemented])
            results_complemented_df = results_complemented_df.append(pd.Series({"model": "USBCF_coclust", "minsim": simthreshold, "k":k, "fold": fold, "rmse": accuracy.rmse(usbcf_complemented),
                                                                                "mae": accuracy.mae(usbcf_complemented),
                                                                               "coverage": len(usbcf_complemented)/preds_usbcf_size,
                                                                                "cov_user": len(users_considered)/len(all_test_users),
                                                                                "cov_item": len(items_considered)/len(all_test_items)}), ignore_index=True)
    
            preds_itembased, _ = dump.load(path_output_useritembased + "iteration" + str(fold) + "/"+"ItemBasedMeans(k=" + str(k) + ",sim=pearson).bpk")
            usbcf_complemented = real_preds_usbcf.copy()
            real_preds_itembased = list(
                filter(lambda x: not x[4]["was_impossible"], preds_itembased))
            itembased_impossible = []
            for pred_itembased in real_preds_itembased:
                impossible_flag = False
                for pred_usbcf in impossible_usbcf_preds:
                    if pred_itembased[0] == pred_usbcf[0] and pred_itembased[1] == pred_usbcf[1]:
                        impossible_flag = True
                        break
                if impossible_flag:
                    itembased_impossible.append(pred_itembased)
            
            #free memory
            preds_itembased = None
            usbcf_complemented.extend(itembased_impossible)
            users_considered = set([pred[0] for pred in usbcf_complemented])
            items_considered = set([pred[1] for pred in usbcf_complemented])
    
            results_complemented_df = results_complemented_df.append(pd.Series({"model": "USBCF_itembased", "minsim": simthreshold, "k":k, "fold": fold, "rmse": accuracy.rmse(usbcf_complemented),
                                                                               "mae": accuracy.mae(usbcf_complemented),
                                                                                "coverage": len(usbcf_complemented)/preds_usbcf_size,
                                                                                "cov_user": len(users_considered)/len(all_test_users),
                                                                                "cov_item": len(items_considered)/len(all_test_items)}), ignore_index=True)

usbcfcomplemented_mean_acc_results = results_complemented_df.groupby(
    ["model", "minsim", "k"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].mean().reset_index()
usbcfcomplemented_std_acc_results = results_complemented_df.groupby(
    ["model", "minsim", "k"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].std().reset_index()

# # usbcf_groupedmincols = usbcf_mean_acc_results.groupby(
# #     ["mincols"])[["mincols", "coverage", "mae", "rmse"]].mean()
# # usbcf_mean_acc_results[usbcf_mean_acc_results["minsim"] == 0.2][[
# #     "mincols", "coverage", "mae", "rmse"]].sort_values(by="mincols")

# # preds_bbcf, _ = dump.load(path_output + "iteration1/" +
# #                           "BBCF(250,QUBIC2(True,100000,10,1,1)).bpk")
# # real_preds_bbcf = list(
# #     filter(lambda x: not x[4]["was_impossible"], preds_bbcf))

# # preds_usbcf, _ = dump.load(
# #     path_output + "iteration1/" + "USBCF(0.25,QUBIC2(True,100000,10,1,1)).bpk")
# # real_preds_usbcf = list(
# #     filter(lambda x: not x[4]["was_impossible"], preds_usbcf))

# # preds_intercept_bbcf = []
# # preds_intercept_usbcf = []
# # for pred_bbcf in real_preds_bbcf:
# #     for pred_usbcf in real_preds_usbcf:
# #         if pred_bbcf[0] == pred_usbcf[0] and pred_bbcf[1] == pred_usbcf[1]:
# #             preds_intercept_bbcf.append(pred_bbcf)
# #             preds_intercept_usbcf.append(pred_usbcf)

# # bbcf_intercept_score = accuracy.rmse(preds_intercept_bbcf)
# # usbcf_intercept_score = accuracy.rmse(preds_intercept_usbcf)


# # #Benchmark results

acc_results_df_benchmarks, speed_results_df_benchmarks = load_results_todf(
    path_output_useritembased)
acc_results_df_benchmarks.columns = ["model", "fold", "coverage", "cov_user",
                                     "cov_item", "mae", "rmse"]

mean_acc_results_df_benchmarks = acc_results_df_benchmarks.groupby(
    ["model"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].mean().reset_index()
std_acc_results_df_benchmarks = acc_results_df_benchmarks.groupby(
    ["model"])[["coverage", "cov_user", "cov_item", "mae", "rmse"]].std().reset_index()
mean_speed_results_df = speed_results_df_benchmarks.groupby(
    ["model"])[["fit_time", "pred_time"]].mean().reset_index()

baseandcoclust_mean_acc_results = mean_acc_results_df_benchmarks[(mean_acc_results_df_benchmarks["model"].str.startswith(
    "baseline")) | (mean_acc_results_df_benchmarks["model"].str.startswith(
        "Coclust"))]
baseandcoclust_std_acc_results = std_acc_results_df_benchmarks[(std_acc_results_df_benchmarks["model"].str.startswith(
    "baseline")) | (std_acc_results_df_benchmarks["model"].str.startswith(
        "Coclust"))]
baseandcoclust_mean_speed_results = mean_speed_results_df[(mean_speed_results_df["model"].str.startswith(
    "baseline")) | (mean_speed_results_df["model"].str.startswith(
        "Coclust"))]

useritembased_mean_acc_results = mean_acc_results_df_benchmarks[(mean_acc_results_df_benchmarks["model"].str.startswith(
    "Item")) | (mean_acc_results_df_benchmarks["model"].str.startswith(
        "User"))]
useritembased_std_acc_results = std_acc_results_df_benchmarks[(std_acc_results_df_benchmarks["model"].str.startswith(
    "Item")) | (std_acc_results_df_benchmarks["model"].str.startswith(
        "User"))]
useritembased_mean_speed_results = mean_speed_results_df[(mean_speed_results_df["model"].str.startswith(
    "Item")) | (mean_speed_results_df["model"].str.startswith(
        "User"))]


useritembased_mean_acc_results["k"] = useritembased_mean_acc_results["model"].str.extract(
    r'(k=\d+)')
useritembased_mean_acc_results["sim"] = useritembased_mean_acc_results["model"].str.extract(
    r'(sim=[a-zA-Z]+)')
useritembased_mean_acc_results["model"] = useritembased_mean_acc_results["model"].apply(
    lambda x: x.split("(")[0])
useritembased_mean_acc_results["k"] = useritembased_mean_acc_results["k"].apply(
    lambda x: float(x.split("=")[-1]))
useritembased_mean_acc_results["sim"] = useritembased_mean_acc_results["sim"].apply(
    lambda x: x.split("=")[-1])

useritembased_std_acc_results["k"] = useritembased_std_acc_results["model"].str.extract(
    r'(k=\d+)')
useritembased_std_acc_results["sim"] = useritembased_std_acc_results["model"].str.extract(
    r'(sim=[a-zA-Z]+)')
useritembased_std_acc_results["model"] = useritembased_std_acc_results["model"].apply(
    lambda x: x.split("(")[0])
useritembased_std_acc_results["k"] = useritembased_std_acc_results["k"].apply(
    lambda x: float(x.split("=")[-1]))
useritembased_std_acc_results["sim"] = useritembased_std_acc_results["sim"].apply(
    lambda x: x.split("=")[-1])

useritembased_mean_speed_results["k"] = useritembased_mean_speed_results["model"].str.extract(
    r'(k=\d+)')
useritembased_mean_speed_results["sim"] = useritembased_mean_speed_results["model"].str.extract(
    r'(sim=[a-zA-Z]+)')
useritembased_mean_speed_results["model"] = useritembased_mean_speed_results["model"].apply(
    lambda x: x.split("(")[0])
useritembased_mean_speed_results["k"] = useritembased_mean_speed_results["k"].apply(
    lambda x: float(x.split("=")[-1]))
useritembased_mean_speed_results["sim"] = useritembased_mean_speed_results["sim"].apply(
    lambda x: x.split("=")[-1])


svd_mean_acc_results = mean_acc_results_df_benchmarks[(mean_acc_results_df_benchmarks["model"].str.startswith(
    "algo_svd")) | (mean_acc_results_df_benchmarks["model"].str.startswith(
        "algo_nmf"))]
svd_std_acc_results = std_acc_results_df_benchmarks[(mean_acc_results_df_benchmarks["model"].str.startswith(
    "algo_svd")) | (std_acc_results_df_benchmarks["model"].str.startswith(
        "algo_nmf"))]
svd_mean_speed_results = mean_speed_results_df[(mean_acc_results_df_benchmarks["model"].str.startswith(
    "algo_svd")) | (std_acc_results_df_benchmarks["model"].str.startswith(
        "algo_nmf"))]

svd_mean_acc_results["n_factors"] = svd_mean_acc_results["model"].str.findall(
    r'(\d*\.?\d+)')
svd_mean_acc_results["n_epochs"] = svd_mean_acc_results["n_factors"].apply(
    lambda x: x[1])
svd_mean_acc_results["reg_all"] = svd_mean_acc_results["n_factors"].apply(
    lambda x: x[2])
svd_mean_acc_results["n_factors"] = svd_mean_acc_results["n_factors"].apply(
    lambda x: x[0])
svd_mean_acc_results["model"] = svd_mean_acc_results["model"].apply(
    lambda x: x.split("-")[0])

svd_std_acc_results["n_factors"] = svd_std_acc_results["model"].str.findall(
    r'(\d*\.?\d+)')
svd_std_acc_results["n_epochs"] = svd_std_acc_results["n_factors"].apply(
    lambda x: x[1])
svd_std_acc_results["reg_all"] = svd_std_acc_results["n_factors"].apply(
    lambda x: x[2])
svd_std_acc_results["n_factors"] = svd_std_acc_results["n_factors"].apply(
    lambda x: x[0])
svd_std_acc_results["model"] = svd_std_acc_results["model"].apply(
    lambda x: x.split("-")[0])

svd_mean_speed_results["n_factors"] = svd_mean_speed_results["model"].str.findall(
    r'(\d*\.?\d+)')
svd_mean_speed_results["n_epochs"] = svd_mean_speed_results["n_factors"].apply(
    lambda x: x[1])
svd_mean_speed_results["reg_all"] = svd_mean_speed_results["n_factors"].apply(
    lambda x: x[2])
svd_mean_speed_results["n_factors"] = svd_mean_speed_results["n_factors"].apply(
    lambda x: x[0])
svd_mean_speed_results["model"] = svd_mean_speed_results["model"].apply(
    lambda x: x.split("-")[0])

svd_elite_mean_acc_results = pd.DataFrame(
    columns=["model", "coverage", "cov_user", "cov_item", "mae", "rmse", "n_factors", "n_epochs", "reg_all"])
svd_elite_std_acc_results = pd.DataFrame(
    columns=["model", "coverage", "cov_user", "cov_item", "mae", "rmse", "n_factors", "n_epochs", "reg_all"])
svd_elite_mean_speed_results = pd.DataFrame(
    columns=["model", "coverage", "cov_user", "cov_item", "mae", "rmse", "n_factors", "n_epochs", "reg_all"])
for group_name, group in svd_mean_acc_results.groupby(["model"]):
    best_model = group.sort_values(by=["mae"]).iloc[0]
    svd_elite_mean_acc_results = svd_elite_mean_acc_results.append(best_model)
    best_model_std = svd_std_acc_results[(svd_std_acc_results.model == best_model.model) & (
        svd_std_acc_results.n_factors == best_model.n_factors) & (
        svd_std_acc_results.n_epochs == best_model.n_epochs) & (svd_std_acc_results.reg_all == best_model.reg_all)]
    svd_elite_std_acc_results = svd_elite_std_acc_results.append(
        best_model_std)
    #best_model_speed = svd_mean_speed_results[(svd_mean_speed_results.model == best_model.model) & (
    #    svd_mean_speed_results.n_factors == best_model.n_factors) & (
    #    svd_mean_speed_results.n_epochs == best_model.n_epochs) & (svd_mean_speed_results.reg_all == best_model.reg_all)]
    #svd_elite_mean_speed_results = svd_elite_mean_speed_results.append(
    #    best_model_speed)


def useritem_latextable(df_mean_acc, df_std_acc, df_mean_speed):
    algorithms = [("UserBased", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                  ("ItemBased", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]

    str_latex = ""
    for algo in algorithms:
        algo_name, params = algo
        for param in params:
            str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
                algo_name + "\\\\(k=" + str(param) + ")\end{tabular}}"+"\t\t\t"
            algo_acc_mean = df_mean_acc.loc[(df_mean_acc['model'].str.startswith(algo_name)) &
                                            (df_mean_acc['k'] == param)]
            algo_acc_std = df_std_acc.loc[(df_std_acc['model'].str.startswith(algo_name)) &
                                          (df_std_acc['k'] == param)]
            algo_speed_mean = df_mean_speed.loc[(df_mean_speed['model'].str.startswith(algo_name)) &
                                                (df_mean_speed['k'] == param)]
            for (idxRow, s1), (_, s2), (_, s3) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows(),
                                                      algo_speed_mean.iterrows()):
                str_latex += " & "
                str_latex += format(round(s1.mae, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
                str_latex += format(round(s1.rmse, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
                str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_user*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_item*100, 4), '.2f')
                # str_latex += datetime.datetime.fromtimestamp(
                #     s3.fit_time / 1000).strftime('%H:%M:%S') + " & "
                # str_latex += datetime.datetime.fromtimestamp(
                #     s3.pred_time / 1000).strftime('%H:%M:%S')

                str_latex += "\\\\"
                str_latex += "\n"
        str_latex += "\\midrule \n"
    str_latex = str_latex[:str_latex.rfind('\n')]

    str_latex += "\\bottomrule"
    print(str_latex)


def svd_latextable(df_mean_acc, df_std_acc, df_mean_speed):
    algorithms = ["algo_svd", "algo_svd++", "algo_nmf"]

    str_latex = ""
    for algo in algorithms:
        algo_name = algo
        str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
            algo_name + "\end{tabular}}"+"\t\t\t"
        algo_acc_mean = df_mean_acc.loc[(
            df_mean_acc['model'] == algo_name)]
        algo_acc_std = df_std_acc.loc[(
            df_std_acc['model'] == algo_name)]
        algo_speed_mean = df_mean_speed.loc[(
            df_mean_speed['model'] == algo_name)]
        for (idxRow, s1), (_, s2), (_, s3) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows(),
                                                  algo_speed_mean.iterrows()):
            str_latex += " & "
            str_latex += format(round(s1.mae, 3), '.3f') + \
                " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
            str_latex += format(round(s1.rmse, 3), '.3f') + \
                " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
            str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
            str_latex += format(round(s1.cov_user*100, 4), '.2f') + " & "
            str_latex += format(round(s1.cov_item*100, 4), '.2f')
            # str_latex += datetime.datetime.fromtimestamp(
            #     s3.fit_time / 1000).strftime('%H:%M:%S') + " & "
            # str_latex += datetime.datetime.fromtimestamp(
            #     s3.pred_time / 1000).strftime('%H:%M:%S')

            str_latex += "\\\\"
            str_latex += "\n"
        str_latex += "\\midrule \n"
    str_latex = str_latex[:str_latex.rfind('\n')]

    str_latex += "\\bottomrule"
    print(str_latex)


def basecoclust_latextable(df_mean_acc, df_std_acc, df_mean_speed):
    algorithms = ["Coclust", "baseline"]

    str_latex = ""
    for algo in algorithms:
        algo_name = algo
        str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
            algo_name + "\end{tabular}}"+"\t\t\t"
        algo_acc_mean = df_mean_acc.loc[(
            df_mean_acc['model'].str.startswith(algo_name))]
        algo_acc_std = df_std_acc.loc[(
            df_std_acc['model'].str.startswith(algo_name))]
        algo_speed_mean = df_mean_speed.loc[(
            df_mean_speed['model'].str.startswith(algo_name))]
        for (idxRow, s1), (_, s2), (_, s3) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows(),
                                                  algo_speed_mean.iterrows()):
            str_latex += " & "
            str_latex += format(round(s1.mae, 3), '.3f') + \
                " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
            str_latex += format(round(s1.rmse, 3), '.3f') + \
                " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
            str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
            str_latex += format(round(s1.cov_user*100, 4), '.2f') + " & "
            str_latex += format(round(s1.cov_item*100, 4), '.2f')

            # str_latex += datetime.datetime.fromtimestamp(
            #     s3.fit_time / 1000).strftime('%H:%M:%S') + " & "
            # str_latex += datetime.datetime.fromtimestamp(
            #     s3.pred_time / 1000).strftime('%H:%M:%S')

            str_latex += "\\\\"
            str_latex += "\n"
        str_latex += "\\midrule \n"
    str_latex = str_latex[:str_latex.rfind('\n')]

    str_latex += "\\bottomrule"
    print(str_latex)

# # def bbcf_latextable(df_mean_acc, df_std_acc, df_mean_speed):
# #     algorithms = [("BBCF", [300])]

# #     str_latex = ""
# #     for algo in algorithms:
# #         algo_name, params = algo
# #         for param in params:
# #             str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
# #                 algo_name + "\\\\(minSim=" + str(param) + ")\end{tabular}}"+"\t\t\t"
# #             algo_acc_mean = df_mean_acc.loc[(df_mean_acc['model'].str.startswith(algo_name)) &
# #                                             (df_mean_acc['nnbrs'] == param)]
# #             algo_acc_std = df_std_acc.loc[(df_std_acc['model'].str.startswith(algo_name)) &
# #                                           (df_std_acc['nnbrs'] == param)]
# #             algo_speed_mean = df_mean_speed.loc[(df_mean_speed['model'].str.startswith(algo_name)) &
# #                                                 (df_mean_speed['minsim'] == param)]
# #             for (idxRow, s1), (_, s2), (_, s3) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows(),
# #                                                       algo_speed_mean.iterrows()):
# #                 str_latex += " & "
# #                 str_latex += format(round(s1.mae, 3), '.3f') + \
# #                     " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
# #                 str_latex += format(round(s1.rmse, 3), '.3f') + \
# #                     " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
# #                 str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
# #                 str_latex += datetime.datetime.fromtimestamp(
# #                     s3.fit_time / 1000).strftime('%H:%M:%S') + " & "
# #                 str_latex += datetime.datetime.fromtimestamp(
# #                     s3.pred_time / 1000).strftime('%H:%M:%S')

# #                 str_latex += "\\\\"
# #                 str_latex += "\n"
# #         str_latex += "\\midrule \n"
# #     str_latex = str_latex[:str_latex.rfind('\n')]

# #     str_latex += "\\bottomrule"
# #     print(str_latex)


def usbcf_latextable(df_mean_acc, df_std_acc, df_mean_speed):
    algorithms = [("USBCFComb", [0.1, 0.2, 0.3, 0.4, 0.5])]

    str_latex = ""
    for algo in algorithms:
        algo_name, params = algo
        for param in params:
            str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
                algo_name + "\\\\(minSim=" + str(param) + \
                ")\end{tabular}}"+"\t\t\t"
            algo_acc_mean = df_mean_acc.loc[(df_mean_acc['model'].str.startswith(algo_name)) &
                                            (df_mean_acc['minsim'] == param)]
            algo_acc_std = df_std_acc.loc[(df_std_acc['model'].str.startswith(algo_name)) &
                                          (df_std_acc['minsim'] == param)]
            algo_speed_mean = df_mean_speed.loc[(df_mean_speed['model'].str.startswith(algo_name)) &
                                                (df_mean_speed['minsim'] == param)]
            for (idxRow, s1), (_, s2), (_, s3) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows(),
                                                      algo_speed_mean.iterrows()):
                str_latex += " & "
                str_latex += format(round(s1.mae, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
                str_latex += format(round(s1.rmse, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
                str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_user*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_item*100, 4), '.2f')

                # str_latex += datetime.datetime.fromtimestamp(
                #     s3.fit_time / 1000).strftime('%H:%M:%S') + " & "
                # str_latex += datetime.datetime.fromtimestamp(
                #     s3.pred_time / 1000).strftime('%H:%M:%S')

                str_latex += "\\\\"
                str_latex += "\n"
        str_latex += "\\midrule \n"
    str_latex = str_latex[:str_latex.rfind('\n')]

    str_latex += "\\bottomrule"
    print(str_latex)


def usbcfcomplemented_latextable(df_mean_acc, df_std_acc):
    algorithms = [("USBCF_coclust", [0.1, 0.2, 0.3, 0.4, 0.5]),
                  ("USBCF_itembased", [0.1, 0.2, 0.3, 0.4, 0.5])]

    str_latex = ""
    for algo in algorithms:
        algo_name, params = algo
        for param in params:
            str_latex += "\multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}}" + \
                algo_name + "\\\\(minSim=" + str(param) + \
                ")\end{tabular}}"+"\t\t\t"
            algo_acc_mean = df_mean_acc.loc[(df_mean_acc['model'].str.startswith(algo_name)) &
                                            (df_mean_acc['minsim'] == param)]
            algo_acc_std = df_std_acc.loc[(df_std_acc['model'].str.startswith(algo_name)) &
                                          (df_std_acc['minsim'] == param)]
            for (idxRow, s1), (_, s2) in zip(algo_acc_mean.iterrows(), algo_acc_std.iterrows()):
                str_latex += " & "
                str_latex += format(round(s1.mae, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.mae, 3), '.3f') + " & "
                str_latex += format(round(s1.rmse, 3), '.3f') + \
                    " $\pm$ " + format(round(s2.rmse, 3), '.3f') + " & "
                str_latex += format(round(s1.coverage*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_user*100, 4), '.2f') + " & "
                str_latex += format(round(s1.cov_item*100, 4), '.2f')

                str_latex += "\\\\"
                str_latex += "\n"
        str_latex += "\\midrule \n"
    str_latex = str_latex[:str_latex.rfind('\n')]

    str_latex += "\\bottomrule"
    print(str_latex)


useritem_latextable(useritembased_mean_acc_results,
                    useritembased_std_acc_results, useritembased_mean_speed_results)
basecoclust_latextable(baseandcoclust_mean_acc_results, baseandcoclust_std_acc_results,
                       baseandcoclust_mean_speed_results)
svd_latextable(svd_elite_mean_acc_results,
               svd_elite_std_acc_results, svd_elite_mean_speed_results)
# # bbcf_latextable()
usbcf_latextable(usbcf_mean_acc_results,
                 usbcf_std_acc_results, usbcf_mean_speed_results)
usbcfcomplemented_latextable(
    usbcfcomplemented_mean_acc_results, usbcfcomplemented_std_acc_results)
