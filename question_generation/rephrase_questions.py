import os
import pandas as pd
from openai import OpenAI



# 你的原始 base prompt（保持不变）
BASE_PROMPT = """
I'm designing a benchmark question for HSE (Health, Safety, and Environment). You are help to rewrite a scenario-based question focused on issue spotting in a regulatory context.

Your task is to rewrite the **scenario and the final question sentence only**. The question tests whether the reader can identify the most relevant legal issue based on a realistic workplace situation.

Here is what you must do:
1. Write the scenario in clear, well-organized prose. It should sound like a real internal incident summary or audit finding. Use a **natural, professional, and concise tone**.
2. Do **not** mention any legal statute or regulation by name.
3. Make the issue **non-obvious**: include some context that may relate to incorrect options, creating plausible confusion.
4. Ensure the scenario includes at least one complicating detail (e.g., conflicting records, partial compliance, or unclear accountability).
5. Do **not** change the meaning of the correct answer, and do **not** modify the answer options.
6. Do not remove or alter any factual information such as the location, job roles, entities involved, or the outcome of the incident.

The original question is:
\"\"\"{question}\"\"\"

The answer choices are:
{options}

The correct answer is option {answer}.

Please return only the rewritten scenario and final question sentence.
"""

# 定义 IRAC 类型附加说明（当 IRAC 类型非 "issue_spotting" 时追加说明，"issue_spotting" 时不做附加）
IRAC_INSTRUCTIONS = {
    "issue_spotting": "",  # 使用原始提示，不追加额外说明
    "rule_recall": (
        "\nFor this IRAC type, focus on rewriting the scenario so that it clearly leads the reader to recall the relevant legal rule or regulatory provision. "
        "Ensure that the description naturally emphasizes the factual basis for the legal rule without directly naming any statute."
    ),
    "rule_application": (
        "\nFor this IRAC type, rewrite the scenario to highlight how the identified legal rule should be applied to the facts. "
        "The narrative should emphasize the reasoning process linking the facts to the legal requirements, including complicating factors."
    ),
    "rule_conclusion": (
        "\nFor this IRAC type, rewrite the scenario in a way that guides the reader toward the final legal conclusion. "
        "Integrate the factual chain with hints of the outcome without explicitly stating it, preserving the logical reasoning steps."
    )
}

# 定义各类别的额外背景补充说明（例如 regulation、safety_exam、court_case、video）
CATEGORY_INSTRUCTIONS = {
    "regulation": "This scenario relates to regulatory compliance within a workplace context.",
    "safety_exam": "This scenario relates to on-site safety issues and worker behavior. ",
    "court_case": "This scenario reflects a legal case with disputes over facts and conflicting responsibilities. ",
    "video": "This scenario is based on observed behavior or dialogue from video recordings, and should reconstruct the facts in a narrative style."
}

def build_prompt(question: str, options: str, answer: str, category: str, irac_type: str) -> str:

    irac_instruction = IRAC_INSTRUCTIONS.get(irac_type.lower(), "")
    category_instruction = CATEGORY_INSTRUCTIONS.get(category.lower(), "")
    
    tail = f"""
The original question is:
\"\"\"{question}\"\"\"

The answer choices are:
{options}

The correct answer is option {answer}.

Please return only the rewritten scenario and final question sentence, with no blank lines in between.
"""
    prompt = BASE_PROMPT + irac_instruction + "\n" + category_instruction + "\n" + tail
    return prompt

def rephrase_question(question: str, options: str, answer: str, category: str, irac_type: str, model_name, client) -> str:
    """
    根据题目、选项、正确答案、类别和 IRAC 类型调用 GPT-4o API 生成重写后的文本。
    """
    prompt = build_prompt(question, options, answer, category, irac_type)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def process_file(input_path: str, output_path: str, category: str, irac_type: str, model_name, client):
    """
    处理 CSV 文件，并根据文件类型（"single", "multi", "tf"）生成改写后的题目。
    参数:
      input_path: 输入 CSV 文件路径
      output_path: 输出 CSV 文件路径
      file_type: 文件类型，"single" 表示单选（题干, A, B, C, D, 答案, 解析）;
                 "multi" 表示多选（题干, A, B, C, D, E, 答案, 解析）;
                 "tf" 表示判断（题干, 答案, 解析），此处固定选项 "A: True\nB: False"
      category: 题目的背景类别，例如 "regulation"
      irac_type: IRAC 类型，例如 "rule_conclusion"
    """
    data = pd.read_csv(input_path, header=None)
    rephrase_questions = []
    
    options_A = []
    options_B = []
    options_C = []
    options_D = []
    answers = []
    explanations = []
    for i in range(len(data)):
        question = data.iloc[i, 0]
        options = (
            f"A: {data.iloc[i, 1]}\n"
            f"B: {data.iloc[i, 2]}\n"
            f"C: {data.iloc[i, 3]}\n"
            f"D: {data.iloc[i, 4]}"
        )
        answer = str(data.iloc[i, 5]).strip()
        rephrased = rephrase_question(question, options, answer, category, irac_type, model_name, client)
        if rephrased is None:
            rephrased = ""
        rephrase_questions.append(rephrased)
        options_A.append(data.iloc[i, 1])
        options_B.append(data.iloc[i, 2])
        options_C.append(data.iloc[i, 3])
        options_D.append(data.iloc[i, 4])
        answers.append(answer)
        explanations.append(data.iloc[i, 6] if data.shape[1] > 6 else "")
        # break
    df = pd.DataFrame({
        0: rephrase_questions,
        1: options_A,
        2: options_B,
        3: options_C,
        4: options_D,
        5: answers,
        6: explanations
    })
        
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, header=False)
    print(f"Rephrased questions saved to {output_path}")



def rephrase_main(config):
    model_name = config.get('model_name', 'gpt-4o')
    openai_api_key = config.get('OPENAI_API_KEY')
    base_url = config.get('OPENAI_BASE_URL')
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=base_url,
    )

    categories = ["regulation","court_case",  "safety_exam", "video"]
    irac_types = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]
    input_paths = [config.get('regulation_output_folder'), config.get('court_case_output_folder'), config.get('exam_output_folder'), config.get('video_output_folder')]
    output_paths = [config.get('regulation_misleading_output_folder'), config.get('court_case_misleading_output_folder'), config.get('exam_misleading_output_folder'), config.get('video_misleading_output_folder')]
    for CATEGORY, input_path, output_path in zip(categories, input_paths, output_paths):
        for IRAC_TYPE in irac_types:
            input_path_single = os.path.join(input_path, f"{IRAC_TYPE}/m1.csv")
            output_path_single = os.path.join(output_path, f"{IRAC_TYPE}/m1.csv")
            process_file(input_path_single, output_path_single, CATEGORY, IRAC_TYPE, model_name, client)
