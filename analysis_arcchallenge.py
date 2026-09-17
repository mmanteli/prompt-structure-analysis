#import plotly.graph_objects as go
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from scipy import stats
from iso639 import languages


base_path = "results"
score_path = "results"
template = "Instruct-Query"
models = [
          #"BAAI__bge-m3",
          #"codefuse-ai__F2LLM-v2-4B",
          #"google__embeddinggemma-300m",
          #"intfloat__multilingual-e5-large-instruct",
          #"microsoft__harrier-oss-v1-0.6b",
          #"Octen__Octen-Embedding-8B",
          "Qwen__Qwen3-Embedding-0.6B",
          #"Qwen__Qwen3-Embedding-4B",
          "__scratch__project_462001491__jmnybl__final_embedding_model_checkpoints__v2-20260909-final__final-finetuned-model"
          ]
dataset = "squad"
split = "dev"
path = lambda model: f"{base_path}/{model}/{dataset}/{split}/{template}_template/"
path_scores = lambda model: f"{score_path}/{model}/{dataset}/{split}/{template}_template/"


def prompts_retrieval():
    return ["Given a question, retrieve the passage that best answers it.",
            "Retrieve.",
            "Find the most relevant passage that directly answers the question.",
            "Given a question, find a related document.",
            "Retrieve the answer to the question.",
            "Retrieve text based on user query.",
            "Given a question, retrieve Wikipedia passages that answer the question.",
            ]
prompts_appropriate = prompts_retrieval



def construct_df(model, show=False, score= "recall@1", 
            displ="sim_q2pq", sim_inc="sim_improvement", angul="chord_similarity",
            columns_to_select=[
                    "score", 
                    "score_distracted", 
                    "score_normalized", 
                    "score_distracted_normalized", 
                    "prompt_text", 
                    "appropriate",
                    "displacement",
                    "sim_improvement",
                    "angulation",
                    "displacement_normalized",
                    "sim_improvement_normalized",
                    "angulation_normalized",
                    ]
                ):
    scores_path= path_scores(model)+f"eval@1_2_5_10.json"
    with open(scores_path) as f:
        scores = json.load(f)
    scores_path2= path_scores(model)+f"eval@1_2_5_10_with_distractors.json"
    with open(scores_path2) as f:
        scores2 = json.load(f)
    df_scores = pd.DataFrame.from_dict(scores).T
    df_scores2 = pd.DataFrame.from_dict(scores2).T
    # there's one prompt twice due to language selection!
    df_scores = df_scores.drop_duplicates(subset="prompt_text")  
    df_scores2 = df_scores2.drop_duplicates(subset="prompt_text")
    # check that the prompt texts match
    # first that they are the same set
    assert set(df_scores["prompt_text"].tolist()) == set(df_scores2["prompt_text"].tolist())
    # and that they're equal length
    assert len(df_scores2) == len(df_scores)
    df_all_scores = df_scores.merge(df_scores2, suffixes=("", "_distracted"), on='prompt_text')
    df = df_all_scores
    df["appropriate"] = [int(p in prompts_appropriate()) for p in df["prompt_text"]]
    df["score"] = [d["mean"] for d in df[score]]
    df["score_distracted"] = [d["mean"] for d in df[f"{score}_distracted"]]
    df["score_normalized"] = stats.zscore(df["score"])
    df["score_distracted_normalized"] = stats.zscore(df["score_distracted"])

    # now read the geometry stuff:
    with open(path(model)+f"prompt_geometry_10nn_1_distractor_and_1_false_positive.json") as f:
        data1 = json.load(f)
    df_geom = pd.DataFrame.from_dict(data1).T
    df_geom = df_geom.drop_duplicates(subset="prompt_text")   # again, one calculated twice!
    for column_name, column in zip(["displacement", "sim_improvement", "angulation"],[displ, sim_inc, angul]):
        df_geom[column_name] = [d["mean"] for d in df_geom[column]]
        df_geom[column_name+"_normalized"] = stats.zscore(df_geom[column_name])
     
    # merge ALL
    df = df.merge(df_geom, on='prompt_text')
    #drop the no-prompt option: metrics do not apply to it
    df = df[df.prompt_text != "NO_PROMPT"]
    if show: display(df.head())
    if columns_to_select:
        return df[columns_to_select]
    return df


# parse results

dfs = {}
score="recall@1"
for m in models:
    dfs_same_model =[]
    try:
        df = construct_df(m, score=score)
        dfs_same_model.append(df)
    except Exception as e:
        print(f"Cannot construct results for {m}")
        raise(e)
    dfs[m] = pd.concat(dfs_same_model)
    print(m)
    print(dfs[m].head())




def score_vs_approriate_analysis(dfs, formula="score_normalized ~ appropriate", 
                                MV=("score_normalized", "appropriate"), 
                                topk=("score_normalized", "appropriate")
                                ):

    full_results = {}
    for model_name, df in dfs.items():
        full_results[model_name] = {}
        # linear regression statsmodels style
        #result_ols = smf.ols(formula, data=df).fit()
        #full_results[model_name]["r2"] = result_ols.rsquared
        # linear regression sklearn style
        score_column, label_column = formula.split(" ~ ")
        scores = np.array(df[score_column])
        prompt_labels = np.array(df[label_column])
        lreg = LinearRegression().fit(prompt_labels.reshape(-1, 1), scores.reshape(-1, 1))
        reg_result = lreg.score(prompt_labels.reshape(-1, 1), scores.reshape(-1, 1))  # r2
        full_results[model_name]["r2"] = reg_result
        # this assert was already checked to hold
        #assert np.isclose(result_ols.rsquared, reg_result), f"{result_ols.rsquared} != {reg_result}" 
        
        if MV:
            full_results[model_name]["r_rb"] = MW_effect_size(df, *MV)
        if topk:
            full_results[model_name]["topk"] = top_k(df, *topk)
    return full_results

def top_k(df, score_column, label_column):
    scores = np.array(df[score_column])
    prompt_labels = np.array(df[label_column])
    appr = scores[prompt_labels == [1]]
    k = len(appr)
    sorted_scores = np.argsort(scores)[::-1][:k]
    best_prompts = np.array([p for p in df["prompt_text"]])[sorted_scores]
    best_prompts_appropriateness = prompt_labels[sorted_scores]
    #print(best_prompts)   # for sanity check; they make sense
    fraction_of_relevant_in_top = sum(best_prompts_appropriateness)/k
    return fraction_of_relevant_in_top

def MW_effect_size(df, score_column, label_column):
    # Mann-Whitney-U
    scores = np.array(df[score_column])
    prompt_labels = np.array(df[label_column])
    appr = scores[prompt_labels == [1]]
    not_appr = scores[prompt_labels == [0]]
    stat, p = stats.mannwhitneyu(appr, not_appr, alternative='greater')
    r_rb =  (2 * stat) / (len(appr) * len(not_appr)) -1
    return (r_rb, p)


full_results_score_only = score_vs_approriate_analysis(dfs)


def format_number(n):
    if isinstance(n, tuple):
        return f"{np.round(n[0],3)}{'*' if n[1]<0.05 else ''}"
    else:
        return str(np.round(n,3))


def to_latex_rows(results):
    for model_name, r in results.items():
        print(" & ".join([model_name.replace("__", "/"), format_number(r["topk"]), format_number(r["r2"]), format_number(r["r_rb"])]),  "\\\\")

print(score)
to_latex_rows(full_results_score_only)
print("------------------------------------")

# add column to all dfs and analyse

score_to_analyse = "score_normalized"
metric_to_analyse = "displacement_normalized"

for model_name, df in dfs.items():
    # full spearman
    sp_full, p_value_full = stats.spearmanr(df[score_to_analyse], df[metric_to_analyse])
    # appropriate and other prompts
    appr_mask = df["appropriate"] == 1
    non_appr_mask = df["appropriate"] == 0
    sp_appr, p_value_appr = stats.spearmanr(df[appr_mask][score_to_analyse], df[appr_mask][metric_to_analyse])
    sp_nappr, p_value_nappr = stats.spearmanr(df[non_appr_mask][score_to_analyse], df[non_appr_mask][metric_to_analyse])

    # linear fit analysis
    df["interaction"] = df[metric_to_analyse] * df["appropriate"]
    result_ols = smf.ols(f"{score_to_analyse}~ {metric_to_analyse} + appropriate + interaction", data=df).fit()
    print("model:", model_name)
    print("Spearman:", f"{np.round(sp_full, 3)}{'*' if p_value_full<0.05 else ''}")
    print("Spearman (appr):", f"{np.round(sp_appr, 3)}{'*' if p_value_appr<0.05 else ''}")
    print("Spearman (others):", f"{np.round(sp_nappr, 3)}{'*' if p_value_nappr<0.05 else ''}")
    print("Linear fit:")
    print("\tR^2:", np.round(result_ols.rsquared, 3))
    for (n, v), (_, p) in zip(result_ols.params.items(), result_ols.pvalues.items()):
        print(f"\t{n}:{np.round(v,3)}{'*' if p<0.05 else ''}")
    print("")


