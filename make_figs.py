# -*- coding: utf-8 -*-
#
# Copyright 2020 Pietro Barbiero, Alberto Tonda and Giovanni Squillero
# Licensed under the EUPL

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mp


results_file = "./results/overall_results.csv"
results = pd.read_csv(results_file, index_col=0)
results["avg test error"] = 1 - results["avg test score"]
results["sem test error"] = results["sem test score"]
results["size"] = results["samples"] * results["features"]
dataset_set = results[["dataset", "samples", "features", "size"]].drop_duplicates().sort_values("size")
n_rows = int(np.sqrt(len(dataset_set)))
dataset_set.drop([0, 48, 64], axis=0, inplace=True)

res = results.copy()
ok_list = [
    "yeast_ml8",
    "scene",
    "isolet",
    "gina_agnostic",
    "gas-drift",
    "letter",
]
idx = []
for i in range(len(res)):
    if res["dataset"][i] in ok_list:
        idx.append(i)
res = res.iloc[idx]
res["size"] = res["samples"] * res["features"]

plt.figure()
sns.lineplot(x="size", y="avg fit time", data=res, hue="method")
plt.xscale("log")
plt.yscale("log")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
plt.show()

# def is_pareto_efficient_simple(costs):
#     """
#     Find the pareto-efficient points
#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype = bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
#             is_efficient[i] = True  # And keep self
#     return is_efficient
#
#
# plt.figure(figsize=[8, 12])
# for i, (k, dataset) in enumerate(dataset_set.iterrows()):
#     d = results[results["dataset"] == dataset["dataset"]]
#     is_pareto = is_pareto_efficient_simple(d[d["method"] != "NO-FS"][["avg fit time", "avg test error"]].values)
#     current_palette = sns.color_palette(n_colors=d.shape[0])
#     plt.subplot(3, 2, i+1)
#     lines = []
#     labels = []
#     for xi, yi, exi, eyi, method, c in zip(d["avg fit time"], d["avg test error"], d["sem fit time"], d["sem test error"], d["method"], current_palette):
#         l = plt.errorbar(xi, yi, xerr=exi, yerr=eyi, fmt='o', ecolor=c, c=c, alpha=0.5)
#         e = mp.Ellipse((xi, yi), width=exi*2, height=eyi*2, color=c, alpha=0.5)
#         plt.gca().add_patch(e)
#         lines.append(l)
#         labels.append(method)
#     pareto_x = d[d["method"] != "NO-FS"]["avg fit time"][is_pareto].values
#     pareto_y = d[d["method"] != "NO-FS"]["avg test error"][is_pareto].values
#     x_sorted = np.argsort(pareto_x)
#     pareto_x = pareto_x[x_sorted]
#     pareto_y = pareto_y[x_sorted]
#     plt.plot(pareto_x, pareto_y, c='k', ls='--', alpha=0.8)
#     plt.scatter(pareto_x, pareto_y, c='k', marker='*', s=80)
#     plt.title(f'{dataset["dataset"]} [S={dataset["samples"]}, F={dataset["features"]}]')
#     plt.xscale("log")
#     if i >= 4:
#         plt.xlabel("fit time")
#     if i % 2 == 0:
#         plt.ylabel("test error")
# plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
# plt.tight_layout()
# plt.savefig("summary.png")
# plt.savefig("summary.pdf")
# plt.show()
