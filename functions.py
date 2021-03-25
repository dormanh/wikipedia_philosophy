import wikipediaapi

import pandas as pd
import numpy as np
import math
import networkx as nx

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
import seaborn as sns
from colormap import rgb2hex

import os
import pickle
import glob
from tqdm.notebook import tqdm as tqdm

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from statsmodels.api import Logit
from scipy import integrate
import scipy.stats as stat

from operator import or_
from functools import reduce
from itertools import combinations, repeat, chain, permutations



### DATA ACQUISITION


def find_category_members(wiki, category, by_name=True, level=0, max_level=1):

    category_members = (
        wiki.page(f"category:{category}").categorymembers.values()
        if by_name
        else category.categorymembers.values()
    )

    titles = set()

    for c in category_members:

        if c.ns != wikipediaapi.Namespace.CATEGORY:
            titles.add(c.title)

        elif level < max_level:

            titles.update(
                find_category_members(
                    c, by_name=False, level=level + 1, max_level=max_level
                )
            )

    return titles


def find_links(wiki, title):

    return set(
        [
            p.title
            for p in wiki.page(title).links.values()
            if p.ns != wikipediaapi.Namespace.CATEGORY
        ]
    )



### FIGURES


font_paths = [
    os.path.join(
        matplotlib.rcParams["datapath"],
        f"/home/borza/Hanga/NS/fonts/SourceSansPro-{f}.ttf",
    )
    for f in ["Bold", "SemiBold", "Regular"]
]


font_props = {
    k: fm.FontProperties(fname=font_paths[i], size=24 - i * 4)
    for i, k in enumerate(["title", "label", "ticks"])
}


def magma_as_hex(n):

    return [
        rgb2hex(*(np.array(c) * 256).astype(int)) for c in sns.color_palette("magma", n)
    ]


def circle(r, center, n_points):
    return [
        [
            math.cos(2 * math.pi / n_points * x) * r + center[0],
            math.sin(2 * math.pi / n_points * x) * r + center[1],
        ]
        for x in range(n_points)
    ]


def degree_dist(
    degrees_list, title="Degree distribution", labels=["Degree"], save=False
):

    colors = magma_as_hex(len(degrees_list))
    low = np.log10(max(1, min([degrees.min() for degrees in degrees_list])))
    high = max([np.log10(degrees.max()) for degrees in degrees_list])
    x = np.logspace(low, high)
    y = np.zeros(shape=(len(degrees_list), x.shape[0] - 1))

    for idx, degrees in enumerate(degrees_list):

        y[idx, :] = pd.cut(pd.Series(degrees), bins=x).value_counts(normalize=True)

        plt.scatter(
            x[1:],
            y[idx],
            c=colors[idx],
            s=200,
            edgecolors="white",
            linewidths=3,
            label=labels[idx],
            alpha=0.8,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, (10 ** (int(high) + 1)))
    plt.ylim(10 ** int(np.log10(y[np.where(y != 0)].min()) - 1), 1)

    plt.xticks(fontproperties=font_props["ticks"])
    plt.yticks(fontproperties=font_props["ticks"])
    plt.xlabel("k", fontproperties=font_props["label"], labelpad=20)
    plt.ylabel("p(k)", fontproperties=font_props["label"], labelpad=20)
    plt.title(title, fontproperties=font_props["title"], pad=20)

    legend = plt.legend(
        prop=font_props["label"],
        markerscale=1.5,
        loc="center right",
        bbox_to_anchor=(1, 0.7),
        edgecolor="white",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(3)

    plt.show()

    if save:
        plt.savefig(
            "figs/{}".format("_".join(title.lower().split(" "))), bbox_inches="tight"
        )
        

def corr_heatmap(
    df,
    title,
    figsize=(15, 15),
    annot=None,
    vmin=0,
    vmax=1,
    mask=None,
    xrotation=90,
    xlabel=None,
    ylabel=None,
    cbar_nticks=6,
    cbar_pad=0.25,
    save=False,
):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("black")

    sns.heatmap(
        df,
        cmap="magma",
        mask=mask,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        linecolor="white",
        linewidths=min(3, 30 / df.shape[0]),
        annot_kws={"fontproperties": font_props["ticks"]},
        cbar_kws={
            "ticks": np.linspace(vmin, vmax, cbar_nticks),
            "orientation": "horizontal",
            "pad": cbar_pad,
            "aspect": 40,
        },
        xticklabels=[" ".join(c.split("_")) for c in df.columns],
        yticklabels=[" ".join(c.split("_")) for c in df.index],
    )

    cb = ax.collections[0].colorbar
    for t in cb.ax.get_xticklabels():
        t.set_font_properties(font_props["ticks"])
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(2)

    plt.xlabel(xlabel, rotation=0, position=(0, 1), fontproperties=font_props["label"])
    plt.ylabel(ylabel, rotation=0, position=(0, 1), fontproperties=font_props["label"])
    plt.yticks(rotation=0, fontproperties=font_props["ticks"])
    plt.xticks(rotation=xrotation, fontproperties=font_props["ticks"])
    plt.title(
        title,
        fontproperties=font_props["title"],
        pad=20,
    )
    plt.show()

    if save:
        plt.savefig(
            "figs/{}".format("_".join(title.lower().split(" "))), bbox_inches="tight"
        )

        
def fields_bar_chart(fields, attr, title, sort_by=None, add_legend=True, save=False):

    sorted_fields = fields.sort_values(by=attr if not sort_by else sort_by).reset_index(
        drop=True
    )
    y_coords = np.linspace(0, 1, sorted_fields.shape[0])

    fig, ax = plt.subplots()

    ax.barh(
        y_coords,
        sorted_fields[attr],
        color=sorted_fields["color_b"],
        linewidth=3,
        left=0,
        height=0.025,
    )

    for i, y in enumerate(y_coords):

        ax.text(
            -0.3 * sorted_fields[attr].max(),
            y - 0.008,
            sorted_fields.loc[i, "field"],
            font_properties=font_props["label"],
        )

    plt.xticks(fontproperties=font_props["ticks"])
    plt.yticks([])
    plt.ylim(-0.025, 1.025)
    plt.title(f"Ranking by: {title}", font_properties=font_props["title"], pad=20)

    if add_legend:
        
        legend = (
            sorted_fields.groupby("broad_field")[["rel_size", "color_b"]]
            .agg({"rel_size": "sum", "color_b": "max"})
            .sort_values(by="rel_size")
            .pipe(
                lambda df: plt.legend(
                    handles=[
                        Patch(facecolor=c, edgecolor="white", linewidth=3)
                        for c in df["color_b"]
                    ],
                    labels=df.index.tolist(),
                    loc="center right",
                    facecolor=None,
                    edgecolor="white",
                    fancybox=False,
                    prop=font_props["label"],
                )
            )
        )
        
        legend.get_frame().set_linewidth(3)

    if save:
        plt.savefig(
            "figs/{}.png".format("_".join(title.split(" "))), bbox_inches="tight"
        )      
        
        
def plot_reg_metrics(metrics_dict, xlabel, ylabel, title, to_plot="roc", save=False):

    sorted_models = (
        pd.concat(
            [
                pd.DataFrame.from_dict(
                    {k: v[to_plot]["auc"]}, orient="index", columns=["auc"]
                )
                for k, v in metrics_dict.items()
            ]
        )
        .sort_values(by="auc", ascending=False)
        .assign(color=magma_as_hex(len(metrics_dict)))
    )

    for i, r in sorted_models.iterrows():

        plt.plot(
            metrics_dict[i][to_plot]["x"],
            metrics_dict[i][to_plot]["y"],
            c=r["color"],
            linewidth=3,
        )

    plt.plot(
        np.linspace(0, 1),
        np.linspace(0, 1) if to_plot == "roc" else np.linspace(1, 0),
        c="grey",
        linewidth=3,
        label="random",
    )

    plt.xticks(fontproperties=font_props["ticks"])
    plt.yticks(fontproperties=font_props["ticks"])
    plt.xlabel(
        xlabel,
        fontproperties=font_props["label"],
        labelpad=20,
    )
    plt.ylabel(
        ylabel,
        fontproperties=font_props["label"],
        labelpad=20,
    )
    plt.title(title, fontproperties=font_props["title"], pad=20)

    legend = plt.legend(
        handles=[
            Patch(
                edgecolor="black",
                facecolor=c,
                linewidth=2,
            )
            for c in sorted_models["color"]
        ],
        labels=sorted_models.apply(
            lambda r: "{} (auc = {})".format(r.name, round(r["auc"], 2)), axis=1
        ).values.tolist(),
        prop=font_props["ticks"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False,
    )

    plt.show()

    if save:
        plt.savefig(f"figs/{to_plot}_curve.png", bbox_inches="tight")
        

        
### DATA ANALYSIS


class LogisticReg(LogisticRegression):

    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance
    in an attribute self.model, and p-values, z scores and estimated
    errors for each coefficient in

    self.z_scores
    self.p_values
    self.sigma_estimates

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(
        self,
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

    def fit(self, X, y):

        self = super(LogisticReg, self).fit(X, y)

        denom = np.tile(
            2.0 * (1.0 + np.cosh(self.decision_function(X))), (X.shape[1], 1)
        ).T

        F_ij = np.dot((X / denom).T, X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij

        return self
    
    
def logit_model(X, y, class_weights={0: 0.01, 1: 0.99}):

    # fit logistic regression

    logit = LogisticReg(
        max_iter=1000, fit_intercept=False, class_weight=class_weights
    ).fit(X, y)

    # calculate roc and precision-recall curves

    prediction = logit.predict_proba(X)[:, 1]
    false_pos, true_pos, tresholds = metrics.roc_curve(y, prediction)
    auc = metrics.roc_auc_score(y, prediction)
    prec, rec, tresholds = metrics.precision_recall_curve(y, prediction)
    auc_pr = metrics.auc(rec, prec)

    metrics_dict = {
        "roc": {
            "x": false_pos,
            "y": true_pos,
            "auc": auc,
        },
        "p-r": {
            "x": prec,
            "y": rec,
            "auc": auc_pr,
        },
    }

    return {"model": logit, "metrics": metrics_dict}
    
    
    
### NETWORKS


def build_graph(nodes, edges, weighted, directed, print_info=True):

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(nodes)

    if weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)

    if print_info:
        print(nx.info(G))

    return G


def neighbor_connectivity(node, graph, weighted, directed):

    w = "weight" if weighted else None

    if directed:
        arr = np.array(
            [
                graph.in_degree(n, weight=w) + graph.out_degree(n, weight=w)
                for n in list(graph.successors(node)) + list(graph.predecessors(node))
            ]
        )

    else:

        arr = np.array([graph.degree(n, weight=w) for n in list(graph.neighbors(node))])

    if arr.shape[0] == 0:
        return 0
    return arr.mean()


def clustering(node, graph):
    
    return nx.clustering(graph, node)


def grow_canopy(node, graph, current_level=0, max_level=2):
    
    canopy = set(graph.successors(node))

    if current_level < max_level:

        for s in canopy.copy():
            
            canopy.update(
                grow_canopy(
                    s, graph=graph, current_level=current_level + 1, max_level=max_level
                )
            )

    return canopy


def disparity_filter(G, weight="weight"):
    """Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
    Args
        G: Weighted NetworkX graph
    Returns
        Weighted graph with a significance score (alpha) assigned to each edge
    References
        M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    """

    if nx.is_directed(G):  # directed case
        N = nx.DiGraph()
        for u in G:

            k_out = G.out_degree(u)
            k_in = G.in_degree(u)

            if k_out > 1:
                sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    w = G[u][v][weight]
                    p_ij_out = float(np.absolute(w)) / sum_w_out
                    alpha_ij_out = (
                        1
                        - (k_out - 1)
                        * integrate.quad(lambda x: (1 - x) ** (k_out - 2), 0, p_ij_out)[
                            0
                        ]
                    )
                    N.add_edge(u, v, weight=w, alpha_out=float("%.4f" % alpha_ij_out))

            elif k_out == 1 and G.in_degree(G.successors(u)[0]) == 1:
                # we need to keep the connection as it is the only way to maintain the connectivity of the network
                v = G.successors(u)[0]
                w = G[u][v][weight]
                N.add_edge(u, v, weight=w, alpha_out=0.0, alpha_in=0.0)
                # there is no need to do the same for the k_in, since the link is built already from the tail

            if k_in > 1:
                sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    w = G[v][u][weight]
                    p_ij_in = float(np.absolute(w)) / sum_w_in
                    alpha_ij_in = (
                        1
                        - (k_in - 1)
                        * integrate.quad(lambda x: (1 - x) ** (k_in - 2), 0, p_ij_in)[0]
                    )
                    N.add_edge(v, u, weight=w, alpha_in=float("%.4f" % alpha_ij_in))
        return N

    else:  # undirected case
        B = nx.Graph()
        for u in G:
            k = len(G[u])
            if k > 1:
                sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    w = G[u][v][weight]
                    p_ij = float(np.absolute(w)) / sum_w
                    alpha_ij = (
                        1
                        - (k - 1)
                        * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
                    )
                    B.add_edge(u, v, weight=w, alpha=float("%.4f" % alpha_ij))
        return B


def disparity_filter_alpha_cut(G, weight="weight", alpha_t=0.4, cut_mode="or"):
    """Performs a cut of the graph previously filtered through the disparity_filter function.

    Args
    ----
    G: Weighted NetworkX graph

    weight: string (default='weight')
        Key for edge data used as the edge weight w_ij.

    alpha_t: double (default='0.4')
        The threshold for the alpha parameter that is used to select the surviving edges.
        It has to be a number between 0 and 1.

    cut_mode: string (default='or')
        Possible strings: 'or', 'and'.
        It works only for directed graphs. It represents the logic operation to filter out edges
        that do not pass the threshold value, combining the alpha_in and alpha_out attributes
        resulting from the disparity_filter function.


    Returns
    -------
    B: Weighted NetworkX graph
        The resulting graph contains only edges that survived from the filtering with the alpha_t threshold

    References
    ---------
    .. M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    """

    if nx.is_directed(G):  # Directed case:
        B = nx.DiGraph()
        for u, v, w in G.edges(data=True):
            try:
                alpha_in = w["alpha_in"]
            except KeyError:  # there is no alpha_in, so we assign 1. It will never pass the cut
                alpha_in = 1
            try:
                alpha_out = w["alpha_out"]
            except KeyError:  # there is no alpha_out, so we assign 1. It will never pass the cut
                alpha_out = 1

            if cut_mode == "or":
                if alpha_in < alpha_t or alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
            elif cut_mode == "and":
                if alpha_in < alpha_t and alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
        return B

    else:
        B = nx.Graph()  # Undirected case:
        for u, v, w in G.edges(data=True):

            try:
                alpha = w["alpha"]
            except KeyError:  # there is no alpha, so we assign 1. It will never pass the cut
                alpha = 1

            if alpha < alpha_t:
                B.add_edge(u, v, weight=w[weight])
        return B
