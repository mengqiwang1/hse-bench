
import csv

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_utils import *

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


SC_NUM = 5
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm



# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embedding of [CLS] token for sentence embedding (the first token)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def get_majority(texts, metrics):
    if len(texts) == 1:
        return texts[0]
    if metrics == "auc":
        # 1. 输入五段文本 计算每段话的 embedding
        embeddings = []
        for text in texts:
            text_emb = get_bert_embedding(text)
            embeddings.append(text_emb)


        # 4. 用 KMeans 聚成 1 类
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(embeddings)

        # 5. 得到聚类中心
        center = kmeans.cluster_centers_[0]

        # 6. 找离中心最近的文本
        similarities = cosine_similarity([center], embeddings)[0]
        best_idx = np.argmax(similarities)
        best_text = texts[best_idx]

        return best_text
    else:

        counter = Counter(texts)
        most_common_string, count = counter.most_common(1)[0]
        return most_common_string

# ✨ 模型调用函数（使用 OpenAI）
def build_prompt(question, options, metrics):
    if metrics == "accuracy":
        prompt = f"""You are an expert in occupational health and safety regulations.
    
    Question:
    {question}
    
    Options:
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    
    Choose the best answer. Reply only with the letter A, B, C, or D.
    **Do NOT explain your choice in the content. Do NOT repeat the question or options.**
    """
    else:
        prompt = question
    return prompt


def get_prompt(class_name, question_type, prompt_method, metrics):
    file_name = f"../input_files/prompt/{class_name}/{question_type}/m1/{prompt_method}-prompt.txt"
    if metrics == "auc":
        file_name = f"../input_files/prompt/{class_name}/{question_type}/m1/auc_{prompt_method}-prompt.txt"
    if prompt_method == "roe":
        file_name = f"../input_files/prompt/{prompt_method}-prompt.txt"
    for enc in ['utf-8', 'gbk']:
        try:
            with open(file_name, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to decode {file_name} as UTF-8 or GBK")


def llm_evaluate(input_csv, output_csv, metrics):
    df = pd.read_csv(input_csv, header=None, names=[
        "question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "reference"
    ])

    # === 1. 如果输出文件不存在，写入表头 ===
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "reference"]
            for model, info in MODELS.items():
                model_name = info["model"]
                header.append(f"{model_name}_answer")
                header.append(f"{model_name}_correct")
            writer.writerow(header)

    # === 2. 执行逐条推理 + 写入 ===
    with open(output_csv, mode="a", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)

        for idx, row in df.iterrows():
            print(f"🧠 Processing Q{idx + 1}/{len(df)}...")

            question = row["question"]
            options = [row["option_A"], row["option_B"], row["option_C"], row["option_D"]]
            correct = row["correct_answer"].strip().upper()
            prompt = build_prompt(question, options, metrics)
            row_data = [row["question"], *options, correct, row["reference"]]

            results = {}

            def run_model(model, info):
                try:
                    print(f"▶ Testing: {model}")
                    if info["api"] == "openai":
                        output = call_openai(info["model"], prompt)
                    elif info["api"] == "anthropic":
                        output = call_anthropic(info["model"], prompt)
                    elif info["api"] == "google":
                        output = call_google(info["model"], prompt)
                    elif info["api"] == "nebius":
                        output = call_nebius(info["model"], prompt, metrics)
                    else:  # default to deepseek
                        output = call_deepseek(info["model"], prompt)
                except Exception as e:
                    print(f"⚠️ Error testing {model}: {e}")
                    output = "?"
                return model, output

            # === 并行执行所有模型 ===
            with ThreadPoolExecutor() as executor:
                future_to_model = {
                    executor.submit(run_model, model, info): model for model, info in MODELS.items()
                }

                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    output = future.result()[1]
                    results[model] = output

            # === 后处理模型结果 ===
            for model, info in MODELS.items():
                output = results[model]
                if metrics == "accuracy":
                    output = output.strip().upper()
                    model_answer = output if output in ['A', 'B', 'C', 'D'] else "?"
                    is_correct = int(model_answer == correct)
                    row_data.extend([model_answer, is_correct])
                else:
                    if output == "?":
                        row_data.extend([output, -1])
                    else:
                        q_embed = get_bert_embedding(output)
                        sim_scores = []
                        for opt in options:
                            opt_embed = get_bert_embedding(opt)
                            sim_scores.append(cosine_similarity(q_embed, opt_embed))

                        correct_index = ord(correct.upper()) - ord('A')
                        labels = [1 if i == correct_index else 0 for i in range(4)]

                        try:
                            auc = roc_auc_score(labels, sim_scores)
                        except Exception as e:
                            print(f"Error calculating AUC: {e}")
                            auc = -1
                        row_data.extend([output, auc])

            writer.writerow(row_data)
            f_out.flush()



def prompt_evaluate(input_csv, output_csv, metrics, class_name, question_type):
    df = pd.read_csv(input_csv, header=None, names=[
        "question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "reference"
    ])
    deepseek_models = {
        "deepseek-v3": "deepseek-chat",
        "deepseek-r1": "deepseek-reasoner"
    }
    prompt_types = ["zero-shot", "few-shot", "0-cot", "cot", "cot+sc", "roe"]

    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "reference"]
            for model_key in deepseek_models.keys():
                for prompt_version in prompt_types:
                    header.append(f"{model_key}_{prompt_version}_answer")
                    header.append(f"{model_key}_{prompt_version}_correct")
            writer.writerow(header)

    # === 2. 执行逐条推理 + 写入 ===
    with open(output_csv, mode="a", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)

        for idx, row in df.iterrows():
            print(f"🧠 Processing Q{idx + 1}/{len(df)}...")

            question = row["question"]
            options = [row["option_A"], row["option_B"], row["option_C"], row["option_D"]]
            correct = row["correct_answer"].strip().upper()
            prompt = build_prompt(question, options, metrics)
            row_data = [row["question"], *options, correct, row["reference"]]

            # 保存结果
            results = {}

            def run_deepseek(model_key, model_name, prompt, prompt_version):
                temp = 0
                sc = 1
                if prompt_version == "cot+sc":
                    temp = 0.7
                cur_prompt = prompt
                if prompt_version in ["cot", "few-shot", "roe"]:
                    cur_prompt = prompt + get_prompt(class_name, question_type, prompt_version, metrics)
                elif prompt_version in ["0-cot"]:
                    cur_prompt = prompt + " Let's think step by step:"
                elif prompt_version in ["cot+sc"]:
                    cur_prompt = prompt + get_prompt(class_name, question_type, "cot", metrics)
                    sc = SC_NUM

                sc_answers = []

                def single_call():
                    try:
                        print(f"▶ Testing: {model_key} [{prompt_version}]")
                        output = call_deepseek(model_name, cur_prompt, temp)
                    except Exception as e:
                        print(f"⚠️ Error testing {model_key} [{prompt_version}]: {e}")
                        output = "?"
                    if metrics == "accuracy":
                        output = output.strip().upper()
                        model_answer = output if output in ['A', 'B', 'C', 'D'] else "?"
                        return model_answer
                    else:
                        return output

                # === 并发跑 sc 次 call_deepseek ===
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(single_call) for _ in range(sc)]
                    for future in as_completed(futures):
                        model_answer = future.result()
                        sc_answers.append(model_answer)

                res = get_majority(sc_answers, metrics)
                return f"{model_name}_{prompt_version}", res

            with ThreadPoolExecutor() as executor:
                future_to_model = {
                    executor.submit(run_deepseek, model_key, model_name, prompt, prompt_version): (model_key, prompt_version)
                    for model_key, model_name in deepseek_models.items()
                    for prompt_version in prompt_types
                }

                for future in as_completed(future_to_model):
                    model_prompt_name, output = future.result()
                    results[model_prompt_name] = output

            # === 后处理并写入row_data ===
            for model_prompt_name in results:
                output = results[model_prompt_name]
                if metrics == "accuracy":
                    is_correct = int(output == correct)
                    row_data.extend([output, is_correct])
                else:
                    if output == "?":
                        row_data.extend([output, -1])
                    else:
                        q_embed = get_bert_embedding(output)
                        sim_scores = []
                        for opt in options:
                            opt_embed = get_bert_embedding(opt)
                            sim_scores.append(cosine_similarity(q_embed.reshape(1, -1), opt_embed.reshape(1, -1))[0, 0])

                        correct_index = ord(correct.upper()) - ord('A')
                        labels = [1 if i == correct_index else 0 for i in range(4)]

                        try:
                            auc = roc_auc_score(labels, sim_scores)
                        except Exception as e:
                            print(f"Error calculating AUC: {e}")
                            auc = -1
                        row_data.extend([output, auc])

            writer.writerow(row_data)
            f_out.flush()

def compute_accuracy(output_csv, metrics):
    df = pd.read_csv(output_csv)
    print(f"Compute {metrics} for {output_csv}...")
    # 获取所有 "_answer" 结尾的列
    answer_cols = [col for col in df.columns if col.endswith("_answer")]

    # 提取模型名
    models = [col[:-7] for col in answer_cols]

    # 只保留那些有对应 "{model}_correct" 列的模型
    valid_models = [model for model in models if f"{model}_correct" in df.columns]

    # 准备写日志的文件路径
    output_dir = os.path.dirname(output_csv)
    log_file = os.path.join(output_dir, metrics + "_res.txt")

    # 打开文件写入输出
    with open(log_file, "w", encoding="utf-8") as f:
        for model in valid_models:
            answer_col = f"{model}_answer"
            correct_col = f"{model}_correct"

            a = (df[answer_col] != "?").sum()
            if metrics == "accuracy":
                b = df[correct_col].astype(int).sum()
            else:
                b = df[df[correct_col] != -1][correct_col].astype(float).sum()

            ratio = b / a if a > 0 else 0
            line = f"{model}: {b}/{a} = {ratio:.4f}"
            print(line)
            f.write(line + "\n")


def get_last_valid_char(s):
    # 使用正则表达式匹配字母，并逆序查找第一个有效字母
    match = re.search(r'[a-zA-Z](?!.*[a-zA-Z])', s[::-1])
    if match:
        return match.group(0)
    return None


def statistics(base_dir, output_dir, metrics):

    CATEGORIES = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]

    def parse_accuracy_file(file_path):
        model_acc = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"(.+?):\s+[\d.]+/[\d.]+\s+=\s+([\d.]+)", line.strip())
                if match:
                    model = match.group(1).strip()
                    acc = float(match.group(2))
                    model_acc[model] = acc
        return model_acc

    def aggregate_accuracy_results(base_dir, metrics):
        model_to_scores = {}

        for category in CATEGORIES:
            acc_file = os.path.join(base_dir, category, f"{metrics}_res.txt")
            if not os.path.exists(acc_file):
                print(f"⚠️ Missing file: {acc_file}")
                continue

            print(f"📄 Parsing {acc_file}...")
            acc_dict = parse_accuracy_file(acc_file)

            for model, acc in acc_dict.items():
                if model not in model_to_scores:
                    model_to_scores[model] = {}
                model_to_scores[model][category] = acc

        # Build and format DataFrame
        df = pd.DataFrame.from_dict(model_to_scores, orient="index")
        # df = df[CATEGORIES]  # Ensure column order
        df.index.name = "model"
        return df

    # Run aggregation and save
    df_result = aggregate_accuracy_results(base_dir, metrics)
    df_result.to_csv(output_dir, float_format="%.4f")


def evaluate_llms(llm_config):
    metrics_list = ["accuracy", "auc"]
    output_paths = [llm_config.get('regulation_misleading_output_folder'), llm_config.get('court_case_misleading_output_folder'), llm_config.get('exam_misleading_output_folder'), llm_config.get('video_misleading_output_folder')]
    for base_dir in output_paths:
        categories = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]
        for metrics in metrics_list:
            for category in categories:
                input_csv = os.path.join(base_dir, category, "m1.csv")
                output_csv = os.path.join(base_dir, category, f"m1_evaluation_{metrics}.csv")
                llm_evaluate(input_csv, output_csv, metrics)



def evaluate_prompts(llm_config):
    metrics_list = ["accuracy", "auc"]
    output_paths = [llm_config.get('regulation_misleading_output_folder'), llm_config.get('court_case_misleading_output_folder'), llm_config.get('exam_misleading_output_folder'), llm_config.get('video_misleading_output_folder')]
    class_names = ["regulation", "court_case", "safety_exam", "video"]
    categories = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]

    for class_name, base_dir in zip(class_names, output_paths):
        for metrics in metrics_list:
            for category in categories:
                input_csv = os.path.join(base_dir, category, "m1.csv")
                output_csv = os.path.join(base_dir, category, f"m1_prompts_{metrics}.csv")
                prompt_evaluate(input_csv, output_csv, metrics, class_name, category)



def recompute_statistics(llm_config):
    for base_dir in [llm_config.get('regulation_misleading_output_folder'), llm_config.get('court_case_misleading_output_folder'), llm_config.get('exam_misleading_output_folder'), llm_config.get('video_misleading_output_folder')]:
        for metrics in ["accuracy", "auc"]:
            for file_type in ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]:
                csv_path = os.path.join(base_dir, file_type, f"m1_prompts_{metrics}.csv")
                compute_accuracy(csv_path, metrics)
            statistics(base_dir, os.path.join(base_dir, f"aggregated_prompts_{metrics}.csv"), metrics)
        for metrics in ["accuracy", "auc"]:
            for file_type in ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]:
                csv_path = os.path.join(base_dir, file_type, f"m1_evaluation_{metrics}.csv")
                compute_accuracy(csv_path, metrics)
            statistics(base_dir, os.path.join(base_dir, f"aggregated_{metrics}.csv"), metrics)


# Function to load configuration from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)


if __name__ == "__main__":
    llm_config = load_config("config.json")
    evaluate_llms(llm_config)
    evaluate_prompts(llm_config)
    recompute_statistics(llm_config)








