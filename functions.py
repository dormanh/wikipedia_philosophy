import os
import pickle
import glob
import gc
from tqdm.notebook import tqdm as tqdm

import wikipediaapi

import pandas as pd
import numpy as np
import math
import networkx as nx
from scipy import integrate
import scipy.stats as stat

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
import seaborn as sns
from colormap import rgb2hex

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from statsmodels.api import Logit, GLS

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


def get(filename, fileformat="parquet"):

    if fileformat == "parquet":

        return pd.read_parquet(f"output/{filename}.{fileformat}")

    with open(f"output/{filename}.{fileformat}", "rb") as fp:
        return pickle.load(fp)
    

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


def circle(n_points, r=1, center=(0, 0), startangle=0):

    angles = np.linspace(startangle, 360 + startangle, n_points + 1)[:-1]

    return [[np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))] for ang in angles]


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
    plt.ylabel("p(k)", rotation=0, fontproperties=font_props["label"], labelpad=20)
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

    if save:
        plt.savefig(
            "figs/{}".format("_".join(title.lower().split(" "))), bbox_inches="tight"
        )
        
        
def sort_adjacency_matrix(matrix):

    return matrix.apply(lambda c: c / c.sum()).pipe(
        lambda df: df.loc[
            df.apply(lambda r: 1 / (r > 0.05).sum(), axis=1)
            .reset_index()
            .rename(columns={0: "sign_count"})
            .sort_values(by=["sign_count", "field_to"])["field_to"]
            .values,
            :,
        ].pipe(lambda df: df.loc[:, df.index.tolist()])
    )
        

def corr_heatmap(
    df,
    title,
    figsize=(15, 15),
    annot=None,
    vmin=0,
    vmax=1,
    mask=None,
    enhance=False,
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
        df if not enhance else df ** 0.5,
        cmap="magma",
        mask=mask,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        center=(vmax + vmin) / 2,
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
    if enhance:
        cb.set_ticklabels(np.square(np.linspace(vmin, vmax, cbar_nticks)).round(1))
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

    if save:
        plt.savefig(
            "figs/{}".format("_".join(title.lower().split(" "))), bbox_inches="tight"
        )
        
        
ranking_dict = {
    "ext_citations": "number of external citations",
    "rel_ext_use": "relative external use",
    "ext_cit_avg": "external citation average",
    "imp_exp_ratio": "import-export ratio",
}


def plot_ranking(attr, sort_by=None, figsize=(20, 20), legend=False, save=False):

    fig, ax = plt.subplots(figsize=figsize)

    sorted_fields = fields.sort_values(by=sort_by if sort_by else attr)
    y_coords = np.linspace(0, 1, sorted_fields.shape[0])

    ax.barh(
        y_coords,
        sorted_fields[attr],
        color=sorted_fields["color_b"],
        linewidth=3,
        left=0,
        height=0.025,
    )

    if attr == "imp_exp_ratio":
        ax.set_xlim(0.5, 1.3)

    for j, y in enumerate(y_coords):

        xlow, xhigh = ax.get_xlim()

        ax.text(
            xlow - (xhigh - xlow) * 0.2,
            y - 0.008,
            sorted_fields.index[j],
            font_properties=font_props["label"],
        )

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_props["ticks"])

    ax.set_yticks([])
    ax.set_ylim(-0.025, 1.025)
    ax.set_title(
        f"Ranking by {ranking_dict[attr]}",
        font_properties=font_props["title"],
        pad=20,
    )

    if legend:

        legend = (
            sorted_fields.reset_index()
            .groupby("broad_field")[["rel_size", "color_b"]]
            .agg({"rel_size": "sum", "color_b": "max"})
            .sort_values(by="rel_size")
            .pipe(
                lambda df: ax.legend(
                    handles=[
                        Patch(facecolor=c, edgecolor="white", linewidth=3)
                        for c in df["color_b"]
                    ],
                    labels=df.index.tolist(),
                    loc="center",
                    bbox_to_anchor=(0.7, 0.5),
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
            "figs/rangsor_{}.png".format("_".join(ranking_dict_hun[attr].split(" "))),
            bbox_inches="tight",
        )
        
        
def plot_reg_metrics(metrics_dict, xlabel, ylabel, title, to_plot="roc", save=False):

    sorted_models = (
        pd.concat(
            [
                pd.DataFrame.from_dict(
                    {k: v["network_attrs"][to_plot]["auc"]},
                    orient="index",
                    columns=["auc"],
                )
                for k, v in metrics_dict.items()
            ]
        )
        .sort_values(by="auc", ascending=False)
        .assign(color=magma_as_hex(len(metrics_dict)))
    )

    for i, r in sorted_models.iterrows():

        for k, v in metrics_dict[i].items():

            plt.plot(
                v[to_plot]["x"],
                v[to_plot]["y"],
                c=r["color"] if k == "network_attrs" else "grey",
                linewidth=3,
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
            for c in np.append(sorted_models["color"].values, "grey")
        ],
        labels=sorted_models.apply(
            lambda r: "{} (auc = {})".format(r.name, round(r["auc"], 2)), axis=1
        ).values.tolist()
        + ["Baseline"],
        prop=font_props["ticks"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False,
    )

    if save:
        plt.savefig(f"figs/{to_plot}_curve.png", bbox_inches="tight")        
        
        
def show_backbone(
    level,
    figsize=(16, 20),
    inner_color="#ffffff",
    facecolor="#313131",
    text_color="#ffffff",
    startangle=0,
    node_scale=1,
    edge_scale=10,
    shrink_factors=(0.2, 0.2),
    text_dist=1.2,
    save=False,
    style="black",
):

    """Main figure"""

    node_layout = {
        **{
            n: circle(fields.shape[0] - 1, startangle=startangle)[i]
            for i, n in enumerate(fields.drop("Philosophy").index)
        },
        **{"Philosophy": (0, 0)},
    }

    text_layout = {
        n: [text_dist * c for c in node_layout[n]] if n != "Philosophy" else (0, -0.1)
        for n in fields.index
    }

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.axis("off")

    nx.draw_networkx_nodes(
        data_dict[f"level_{level}"]["backbone"],
        pos=node_layout,
        node_size=[
            fields.loc[n, "n_nodes_weighted"] * node_scale
            for n in data_dict[f"level_{level}"]["backbone"].nodes()
        ],
        node_color=inner_color,
        edgecolors=[
            fields.loc[n, "color_b"]
            for n in data_dict[f"level_{level}"]["backbone"].nodes()
        ],
        linewidths=3,
    )

    for edge in data_dict[f"level_{level}"]["backbone"].edges(data=True):

        nx.draw_networkx_edges(
            data_dict[f"level_{level}"]["backbone"],
            pos=node_layout,
            edgelist=[edge],
            edge_color=fields.loc[edge[0], "color_b"],
            arrowstyle=matplotlib.patches.ArrowStyle.Wedge(
                tail_width=edge[2]["weight"] * edge_scale * 2,
                shrink_factor=shrink_factors[0],
            ),
            connectionstyle=matplotlib.patches.ConnectionStyle("Arc3", rad=-0.3),
        )

        nx.draw_networkx_edges(
            data_dict[f"level_{level}"]["backbone"],
            pos=node_layout,
            edgelist=[edge],
            edge_color=inner_color,
            arrowstyle=matplotlib.patches.ArrowStyle.Wedge(
                tail_width=edge[2]["weight"] * edge_scale,
                shrink_factor=shrink_factors[1],
            ),
            connectionstyle=matplotlib.patches.ConnectionStyle("Arc3", rad=-0.3),
        )

    text_outline = "#ffffff" if (text_color == "#000000") else "#000000"

    for k, v in text_layout.items():

        ax.annotate(
            k,
            v,
            horizontalalignment="left" if np.sign(node_layout[k][0]) == 1 else "right",
            font_properties=font_props["label"],
            color=text_color,
            fontsize=24 if k == "Philosophy" else 18,
            path_effects=[PathEffects.withStroke(linewidth=3, foreground=text_outline)],
        )

    """Legend"""

    legend_g = nx.Graph()
    legend_g.add_edges_from([[i, i + 1] for i in range(3)])
    legend_layout = {i: (np.linspace(-1, 1, 4)[i], -text_dist - 0.5) for i in range(4)}

    nx.draw_networkx_nodes(
        legend_g,
        pos=legend_layout,
        node_size=1000,
        node_color="#000000",
        edgecolors=["#ffffff" if n <= level else facecolor for n in legend_g.nodes()],
        linewidths=4,
    )

    nx.draw_networkx_edges(
        legend_g,
        pos=legend_layout,
        edgelist=[edge for edge in legend_g.edges() if edge[1] <= level],
        width=30,
        edge_color="#ffffff",
    )
    nx.draw_networkx_edges(legend_g, pos=legend_layout, width=20, edge_color="#000000")

    ax.annotate(
        f"Cross-field edges based on level {level} connections",
        (-1, -text_dist - 0.3),
        font_properties=font_props["label"],
        color=text_color,
        path_effects=[PathEffects.withStroke(linewidth=3, foreground=text_outline)],
    )

    if save:
        plt.savefig(
            f"figs/fancy_backbones/backbone_{level}_{style}.png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

        
### DATA ANALYSIS


def gls_model(X, y, test_size=0.25):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    gls = GLS(y_train, X_train).fit(cov_type='HC1')

    coeffs = (
        gls.params.to_frame()
        .rename(columns={0: "coeff"})
        .merge(
            gls.pvalues.to_frame().rename(columns={0: "p-value"}),
            left_index=True,
            right_index=True,
        )
        .assign(
            coeff=lambda df: df.apply(
                lambda r: r["coeff"] if r["p-value"] < 0.05 else np.nan, axis=1
            )
        )[["coeff"]]
    )

    mse = gls.predict(X_test).pipe(lambda s: (s - y_test) ** 2).mean()

    return {"model": gls, "coeffs": coeffs, "mse": mse}


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
    
    
def logit_model(X, y, test_size=0.25):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    logit = LogisticReg(
        max_iter=1000,
        fit_intercept=False,
    ).fit(X_train, y_train)

    prediction = logit.predict_proba(X_test)[:, 1]
    false_pos, true_pos, roc_thresholds = metrics.roc_curve(y_test, prediction)
    auc = metrics.roc_auc_score(y_test, prediction)
    prec, rec, pr_thresholds = metrics.precision_recall_curve(y_test, prediction)
    auc_pr = metrics.auc(rec, prec)

    metrics_dict = {
        "roc": {
            "x": false_pos,
            "y": true_pos,
            "th": roc_thresholds,
            "auc": auc,
        },
        "p-r": {
            "x": rec,
            "y": prec,
            "th": pr_thresholds,
            "auc": auc_pr,
        },
    }

    return {"model": logit, "metrics": metrics_dict}


def gini(x):

    return np.concatenate([[abs(x_i - x_j) for x_j in x] for x_i in x]).sum() / (
        2 * x.shape[0] ** 2 * x.mean()
    )
    
    
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


def directed_neighbor_connectivity(node, graph, direction):

    if direction not in {"in", "out", "both"}:
        raise ValueError("Invalid direction. Expected 'in', 'out' or 'both'.")

    if direction == "in":
        arr = np.array([graph.in_degree(n) for n in graph.predecessors(node)])

    elif direction == "out":
        arr = np.array([graph.out_degree(n) for n in graph.successors(node)])

    else:
        arr = np.array(
            [
                graph.in_degree(n) + graph.out_degree(n)
                for n in list(graph.successors(node)) + list(graph.predecessors(node))
            ]
        )

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


def backbone_extraction(G, alpha, directed=True, print_info=True):

    N = nx.DiGraph() if directed else nx.Graph()

    for n in G.nodes():

        if directed:
            edges_list = [G.out_edges(n, data=True), G.in_edges(n, data=True)]

        else:
            edges_list = [G.edges(n, data=True)]

        significant_edges = [
            (
                pd.DataFrame({edge[1 - i]: edge[2] for edge in edges})
                .T.assign(
                    rel_weight=lambda df: df["weight"].pipe(lambda s: s / s.sum())
                )
                .pipe(
                    lambda df: df.assign(
                        alpha=df["rel_weight"].apply(
                            lambda w: (
                                1
                                - (len(edges) - 1)
                                * integrate.quad(
                                    lambda x: (1 - x) ** (len(edges) - 2), 0, w
                                )[0]
                            )
                        )
                    )
                )
                .loc[lambda df: df["alpha"] < alpha]
            )["weight"].to_dict()
            if len(edges) > 1
            else {edge[1 - i]: edge[2]["weight"] for edge in edges}
            for i, edges in enumerate(edges_list)
        ]

        for i, edges in enumerate(significant_edges):

            if edges:

                N.add_edges_from(
                    [
                        (n, k, {"weight": v}) if (i == 0) else (k, n, {"weight": v})
                        for k, v in edges.items()
                    ]
                )

    if print_info:
        print(nx.info(N))

    return N


def depth_n_field_edges(G, n, index_df, batchsize=100_000):

    # get adjacency matrix
    adj_sparse = nx.adjacency_matrix(G) ** n
    npedges = np.zeros((adj_sparse.data.shape[0], 3), dtype=int)

    for i, rowstart in enumerate(adj_sparse.indptr[:-1]):
        rowend = adj_sparse.indptr[i + 1]
        npedges[rowstart:rowend, 0] = i
        npedges[rowstart:rowend, 1] = adj_sparse.indices[rowstart:rowend]
        npedges[rowstart:rowend, 2] = adj_sparse.data[rowstart:rowend]

    edges = pd.DataFrame(npedges, columns=["from", "to", "weight"])
    #edges.to_parquet(f"output/depth_{n}_edges.parquet")

    # collect garbage
    del adj_sparse
    gc.collect()

    # separate single and multiple field nodes
    (_, singlefield), (__, multifield) = index_df.groupby(index_df["fieldcount"] > 1)

    # collect data of single field nodes
    indsets = {f: gdf["index"].pipe(set) for f, gdf in singlefield.groupby("field")}

    slinks = []

    for gid, inds in tqdm(indsets.items()):

        targs = edges.loc[lambda df: df["from"].isin(inds), ["to", "weight"]]

        for gid2, inds2 in indsets.items():
            slinks.append(
                {
                    "field_from": gid,
                    "field_to": gid2,
                    "weight": targs.loc[
                        lambda df: df["to"].isin(inds2), "weight"
                    ].sum(),
                }
            )

    field_edges = pd.DataFrame(slinks).set_index(["field_from", "field_to"])

    # collect data of multiple field nodes
    multi_inds = multifield["index"].pipe(set)
    multi_edges = edges.loc[
        lambda df: df["from"].isin(multi_inds) | df["to"].isin(multi_inds), :
    ]

    for i in tqdm(range(0, multi_edges.shape[0], batchsize)):

        batch_links = (
            multi_edges.iloc[i : (i + batchsize), :]
            .merge(index_df[["index", "field"]], left_on="from", right_on="index")
            .drop("index", axis=1)
            .rename(columns={"field": "field_from"})
            .merge(index_df[["index", "field"]], left_on="to", right_on="index")
            .drop("index", axis=1)
            .rename(columns={"field": "field_to"})
            .assign(
                weight=lambda df: df["weight"]
                / df.groupby(["from", "to"])["weight"].transform("count")
            )
            .drop(["from", "to"], axis=1)
            .groupby(["field_from", "field_to"])
            .sum()
        )
        ind_union = field_edges.index.union(batch_links.index)
        field_edges = field_edges.reindex(ind_union).fillna(0) + batch_links.reindex(
            ind_union
        ).fillna(0)

    field_edges.to_parquet(f"output/depth_{n}_field_edges.parquet")
    field_link_matrix = field_edges.reset_index().pivot_table(
        index="field_to", columns="field_from", values="weight"
    )
    field_link_matrix.to_parquet(f"output/depth_{n}_field_link_matrix.parquet")

    return field_link_matrix
