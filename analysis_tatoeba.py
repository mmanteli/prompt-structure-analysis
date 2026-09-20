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
          "BAAI__bge-m3",
          "codefuse-ai__F2LLM-v2-4B",
          "google__embeddinggemma-300m",
          "intfloat__multilingual-e5-large-instruct",
          "microsoft__harrier-oss-v1-0.6b",
          "Octen__Octen-Embedding-8B",
          "Qwen__Qwen3-Embedding-0.6B",
          "Qwen__Qwen3-Embedding-4B",
          #"__scratch__project_462001491__jmnybl__checkpoint-19478-tatoeba"
          ]
dataset = lambda lang: f"mteb__tatoeba-bitext-mining:{lang}" 
split = "test"
path = lambda model, lang: f"{base_path}/{model}/{dataset(lang).replace(':','_')}/{split}/{template}_template/"
path_scores = lambda model, lang: f"{score_path}/{model}/{dataset(lang).replace(':','_')}/{split}/{template}_template/"


def prompts_lang(l):
    l, eng = l.split("-")
    assert eng == "eng"
    try:
        lang = languages.get(part2t=l).name
    except KeyError as err:
        if l == "cmn":  # this cannot be parsed by the library
            lang = "Mandarin Chinese"
        else:
            raise(err)
            
    return ["Retrieve parallel sentences.",
            f"Retrieve the corresponding translation in {lang}.",
            f"Given an English sentence, find its translation in {lang}.",
            f"Retrieve parellel sentences in {lang}.",
            f"Translate to {lang}.",
            "Find a sentence that has similar meaning.",  # this is twice here because it was accidentally included twice in the calc
            "Retrieve the corresponding translation.",
            "Given an English sentence, find its translation.",
            "Retrieve parellel sentences.",
            "Translate.",
            "Find a sentence that has similar meaning.",
            ]
prompts_appropriate = prompts_lang  # maps function


def construct_df(model, lang, show=False, score= "recall@1", 
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
    scores_path= path_scores(model, lang)+f"eval@1_2_5_10.json"
    with open(scores_path) as f:
        scores = json.load(f)
    scores_path2= path_scores(model, lang)+f"eval@1_2_5_10_with_distractors.json"
    with open(scores_path2) as f:
        scores2 = json.load(f)
    df_scores = pd.DataFrame.from_dict(scores).T
    df_scores2 = pd.DataFrame.from_dict(scores2).T
    # there's one prompt twice due to language selection!
    df_scores = df_scores.drop_duplicates(subset="prompt_text")  # 
    df_scores2 = df_scores2.drop_duplicates(subset="prompt_text")
    # check that the prompt texts match
    # first that they are the same set
    assert set(df_scores["prompt_text"].tolist()) == set(df_scores2["prompt_text"].tolist())
    # and that they're equal length
    assert len(df_scores2) == len(df_scores)
    df_all_scores = df_scores.merge(df_scores2, suffixes=("", "_distracted"), on='prompt_text')
    df = df_all_scores
    df["appropriate"] = [int(p in prompts_appropriate(lang)) for p in df["prompt_text"]]
    df["score"] = [d["mean"] for d in df[score]]
    df["score_distracted"] = [d["mean"] for d in df[f"{score}_distracted"]]
    df["score_normalized"] = stats.zscore(df["score"])
    df["score_distracted_normalized"] = stats.zscore(df["score_distracted"])

    # now read the geometry stuff:
    with open(path(model, lang)+f"prompt_geometry_10nn_1_distractor_and_1_false_positive.json") as f:
        data1 = json.load(f)
    df_geom = pd.DataFrame.from_dict(data1).T
    df_geom = df_geom.drop_duplicates(subset="prompt_text")   # again, one calculated twice!
    for column_name, column in zip(["displacement", "sim_improvement", "angulation"],[displ, sim_inc, angul]):
        df_geom[column_name] = [d["mean"] for d in df_geom[column]]
        df_geom[column_name+"_normalized"] = stats.zscore(df_geom[column_name])
    

    # merge ALL
    df = df.merge(df_geom, on='prompt_text')
    #df = df[df.prompt_text != "NO_PROMPT"]
    if show: display(df.head())
    if columns_to_select:
        return df[columns_to_select]
    return df


# parse results

dfs = {}
score="recall@1"
for m in models:
    dfs_same_model =[]
    for lang in ["cmn-eng", "fin-eng", "vie-eng", "ara-eng", "tur-eng", "fra-eng", "spa-eng", "deu-eng"]:
        try:
            df = construct_df(m, lang, score=score)
            df["language"] = lang   # add "source" for filtering
            dfs_same_model.append(df)
        except Exception as e:
            print(f"Cannot construct results for {m}")
            raise(e)
    dfs[m] = pd.concat(dfs_same_model)




def mlm_analysis(dfs, analysis_formula, MV=("score_normalized", "appropriate"), topk=("score_normalized", "appropriate")):

    def marginal_r2_mlm(result):
        """Variance explained by the _appropriate label_ only, compared to total variance explained == R^2"""
        # variance explained by the labels 
        # fe params = intercept + appropriate (coefficients for the fit)
        # "use the model" by multiplying by [intercept(always 1), appropriate(0 or 1)]
        fe_predictions = result.model.exog @ result.fe_params
        var_fe = np.var(fe_predictions)
        # random effect variance: estimated already in fitting, .iloc[0,0] only extracts the value
        var_re = float(result.cov_re.iloc[0, 0])
        # Residual variance: estimated in fitting
        var_resid = result.scale
        return var_fe / (var_fe + var_re + var_resid)

    full_results = {}
    for model_name, df in dfs.items():
        full_results[model_name] = {}
        if topk:
            full_results[model_name]["topk"] = top_k(df, *topk)
        model = smf.mixedlm(analysis_formula, df, groups=df["language"], )
        result = model.fit()
        if result.converged:
            full_results[model_name]["r2"] = marginal_r2_mlm(result)
        else:
            print(f"Not converged for {model_name}")
            full_results[model_name]["r2"] = None
        if MV:
            full_results[model_name]["MV"] = MW_effect_size(df, *MV)
    return full_results

def top_k(df, score_column, label_column):
    scores = np.array(df[score_column])
    prompt_labels = np.array(df[label_column])
    appr = scores[prompt_labels == [1]]
    k = len(appr)
    sorted_scores = np.argsort(scores)[::-1][:k]
    best_prompts = np.array([p for p in df["prompt_text"]])[sorted_scores]
    best_prompts_appropriateness = prompt_labels[sorted_scores]
    #print(best_prompts)   # for sanity check
    fraction_of_relevant_in_top = sum(best_prompts_appropriateness)/k
    return fraction_of_relevant_in_top

def MW_effect_size(df, score_column, label_column):
    # top-k and Mann-Whitney-U
    scores = np.array(df[score_column])
    prompt_labels = np.array(df[label_column])
    appr = scores[prompt_labels == [1]]
    not_appr = scores[prompt_labels == [0]]
    stat, p = stats.mannwhitneyu(appr, not_appr, alternative='greater')
    r_rb =  (2 * stat) / (len(appr) * len(not_appr)) -1
    return (r_rb, p)
    

full_results = mlm_analysis(dfs, "score ~ appropriate")

print(full_results)


def format_number(n):
    if isinstance(n, tuple):
        return f"{np.round(n[0],3)}{'*' if n[1]<0.05 else ''}"
    else:
        return str(np.round(n,3))


def to_latex_rows(results, column_names=None):
    if column_names:
        print(" & ".join([c.replace("_"," ") for c in column_names]), "\\\\")
    for model_name, r in results.items():
        print(" & ".join([model_name.replace("__", "/")]+ [format_number(r_) for r_ in r.values()]),  "\\\\")

print(score)
to_latex_rows(full_results)
print("------------------------------------")

# add column to all dfs and analyse


score_to_analyse = "score_normalized"
for metric_to_analyse in ["displacement_normalized", "sim_improvement_normalized", "angulation_normalized"]:

    m_results = {}
    for model_name, df in dfs.items():
        m_results[model_name] = {}
        # full spearman
        sp_full, p_value_full = stats.spearmanr(df[score_to_analyse], df[metric_to_analyse])
        # appropriate and other prompts
        appr_mask = df["appropriate"] == 1
        non_appr_mask = df["appropriate"] == 0
        sp_appr, p_value_appr = stats.spearmanr(df[appr_mask][score_to_analyse], df[appr_mask][metric_to_analyse])
        sp_nappr, p_value_nappr = stats.spearmanr(df[non_appr_mask][score_to_analyse], df[non_appr_mask][metric_to_analyse])

        # linear fit analysis
        df["interaction"] = df[metric_to_analyse] * df["appropriate"]
        result_ols = smf.ols(f"{score_to_analyse} ~ {metric_to_analyse} + appropriate + interaction", data=df).fit()   #
        #print("model:", model_name)
        #print("Spearman:", f"{np.round(sp_full, 3)}{'*' if p_value_full<0.05 else ''}")
        #print("Spearman (appr):", f"{np.round(sp_appr, 3)}{'*' if p_value_appr<0.05 else ''}")
        #print("Spearman (others):", f"{np.round(sp_nappr, 3)}{'*' if p_value_nappr<0.05 else ''}")
        #print("Linear fit:")
        #print("\tR^2:", np.round(result_ols.rsquared, 3))
        #for (n, v), (_, p) in zip(result_ols.params.items(), result_ols.pvalues.items()):
        #    print(f"\t{n}:{np.round(v,3)}{'*' if p<0.05 else ''}")
        #print("")
        m_results[model_name]["spearman_full"] = (sp_full, p_value_full)
        #m_results[model_name]["spearman_appr"] = (sp_appr, p_value_appr)
        #m_results[model_name]["spearman_nappr"] = (sp_nappr, p_value_nappr)
        m_results[model_name]["r2"] = result_ols.rsquared
        for (n, v), (_, p) in zip(result_ols.params.items(), result_ols.pvalues.items()):
            m_results[model_name][n] = (v,p)

    print(metric_to_analyse)
    to_latex_rows(m_results, column_names=["Model"]+[i for i in m_results["BAAI__bge-m3"].keys()])
    print("------------------------------------------")



m_results={}
score_to_analyse = "score_normalized"
for model_name, df in dfs.items():
    m_results[model_name] = {}
    for metric_to_analyse in ["displacement_normalized", "sim_improvement_normalized", "angulation_normalized"]:
        sp_full, p_value_full = stats.spearmanr(df[score_to_analyse], df[metric_to_analyse])
        m_results[model_name][f"spearman_{metric_to_analyse}"] = (sp_full, p_value_full)
print(score_to_analyse)
to_latex_rows(m_results, column_names=["Model"]+[i for i in m_results["BAAI__bge-m3"].keys()])



for model_name, df in dfs.items():
    dist= df["score_distracted"]
    reg = df["score"]
    print(model_name,min(dist),max(dist),max(reg))