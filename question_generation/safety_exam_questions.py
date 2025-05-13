import json
import os
import time
import csv
from openai import OpenAI


def llm_processing(exam_content,  model_name, client):
    with open("../input_files/IRAC_Framework.txt", mode='rb') as file:
        irac_rules = file.read()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
The details about this IRAC framework can be found here: {irac_rules}. I now need to apply this framework to the HSE field in a way that challenges examinees to perform multi-layered legal analysis and critical evaluation.

I have a set of existing safety exam questions about HSE. The content of these questions is: {exam_content}. Note that the specific region applicable to these questions is indicated in the first line of the content.

Based on these safety exam questions, generate scenario-based questions that adhere strictly to the IRAC framework. For each scenario, generate questions addressing four key components:

1. Issue Spotting – Identify the critical safety or legal issue presented in the scenario.
2. Rule Recall – State the relevant legal rule(s) or regulation(s) applicable to the identified issue.
3. Rule Application – Explain in detail how the legal rule applies to the given facts. The explanation must include:
   - Specific inferences drawn from the facts.
   - A detailed reasoning process that connects the facts to the rule.
   - A clear statement of the expected outcome based on these inferences.
4. Rule Conclusion – Present the final legal conclusion derived from the rule application.

For each of these four IRAC components, generate a multiple choice question with only one correct answer (labeled as m1_question)

For every question, provide:
- A correct answer (m1_answer).
- A comprehensive explanation (m1_explanation) that meets the following criteria:
   • Correctness: The explanation must be factually accurate with no misstatements of the legal rule or fact pattern.
   • Analysis: The explanation must include detailed inferences from the facts and demonstrate how the conclusion is reached. It should not be a mere restatement of the rule.
   • Specificity: Every explanation MUST reference a specific, up-to-date HSE regulation (with its name and clause number if applicable) relevant to the given region. Use online search to ensure the legal basis is current.
   • Uniqueness: Each explanation must be unique and tailored to the scenario, avoiding generic or boilerplate language.

Additionally, ensure adversarial generation of answer options by including distractor options:
   - For multiple-choice questions, in addition to the correct answer (m1_answer), also give a trap option (m1_trap_answer) that is roughly half are correct (or partially correct) and half are incorrect. It should seems to be correct for non-expert examinees or LLMs, which correlates to some contexts in the scenario.
   - The distractor options should be plausible, referencing real-world regulatory elements but include subtle errors (e.g., incorrect clause numbers, misinterpretations of the rule, or irrelevant legal principles) to challenge examinees.
   - In the explanations, clearly outline why the correct options are valid and explicitly identify the misleading elements in the distractor options.

In summary, the generated questions should compel the examinee to:
   • Identify nuanced safety issues from scenarios that include misleading or extraneous information.
   • Recall and articulate specific legal rules with awareness of subtle distractors.
   • Apply these rules with detailed legal analysis, critically examining ambiguous or adversarial answer options.
   • Reach a reasoned legal conclusion supported by specific regulatory references.
   • Provide comprehensive explanations that not only justify the correct answer but also systematically deconstruct the distractor options.

Generate the complete set of questions, answers, and explanations for each IRAC component and each question format.
                    """
                }
            ]
        }
    ]

    irac_property = {
        "type": "object",
        "properties": {
            "m1_question": {"type": "string", "description": "multiple choice question body with one correct answer"},
            "m1_option_A": {"type": "string", "description": "option A for m1_question"},
            "m1_option_B": {"type": "string", "description": "option B for m1_question"},
            "m1_option_C": {"type": "string", "description": "option C for m1_question"},
            "m1_option_D": {"type": "string", "description": "option D for m1_question"},
            "m1_answer": {"type": "string", "description": "answer for m1_question", "enum": ["A", "B", "C", "D"]},
            "m1_explanation": {"type": "string", "description": "explanation for the answer of m1_question"},
        },
        "required": ["m1_question", "m1_option_A", "m1_option_B", "m1_option_C", "m1_option_D", "m1_answer", "m1_explanation"],
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "exam_questions",
            "description": "Ask questions about the provided exam questions in different types and modes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "issue_spotting": irac_property,
                    "rule_recall": irac_property,
                    "rule_application": irac_property,
                    "rule_conclusion": irac_property
                },
                "required": ["scenario", "issue_spotting", "rule_recall", "rule_application", "rule_conclusion"],
                "additionalProperties": False
            },
        }
    }]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            tools=tools,
        )
        print(response)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"


def validate_arguments(arguments):
    """
    校验返回的 arguments 是否包含所有必备的 IRAC 部分及其必须的键。
    仅对 IRAC 框架中的组件进行校验；而 "scenario" 字段仅作为字符串，不要求包含 m1_question 等字段。
    返回 True 表示数据完整可用，否则返回 False。
    """
    # "scenario" 只需要存在即可
    if "scenario" not in arguments:
        print("Error: 缺少必备键 scenario")
        return False

    # 仅对 IRAC 部分进行字段校验
    required_components = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]
    required_m1_fields = [
        "m1_question",
        "m1_option_A",
        "m1_option_B",
        "m1_option_C",
        "m1_option_D",
        "m1_answer",
        "m1_explanation"
    ]
    for comp in required_components:
        if comp not in arguments:
            print(f"Error: 缺少必备键 {comp}")
            return False
        for field in required_m1_fields:
            if field not in arguments[comp]:
                print(f"Error: 在 {comp} 中缺少必要字段 {field}")
                return False
    return True



def write_csv(arguments, output_folder, base_filename):
    """
    将每个 IRAC 组件写入对应的 CSV 文件。
    假定数据已经校验完整。
    """
    scenario = arguments.get("scenario", "")
    irac_components = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]
    for question_type in irac_components:
        argument = arguments[question_type]
        question = argument["m1_question"]
        file_name = os.path.join(output_folder, f"{question_type}/m1.csv")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        try:
            with open(file_name, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow((
                    f"{scenario} {question}",
                    argument["m1_option_A"],
                    argument["m1_option_B"],
                    argument["m1_option_C"],
                    argument["m1_option_D"],
                    argument["m1_answer"],
                    argument["m1_explanation"],
                    base_filename
                ))
        except Exception as ex:
            print(f"Error writing CSV for component '{question_type}': {str(ex)}")
            raise  # 若写入失败，则抛出异常以保证一致性


def write_full_answer(arguments, output_folder):
    """
    保存完整答案至一个 txt 文件。
    """
    output_path = os.path.join(output_folder, "full_output.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(arguments, ensure_ascii=False, indent=2))
            f.write("\n\n")
    except Exception as e:
        print(f"Error writing full answer to txt: {str(e)}")
        raise


def read_txt_file(file_path):
    """读取 TXT 文件内容并返回字符串。"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def process_exam_file(file_path, output_folder, model_name, client,scenario_count=8, max_retries=8):
    """
    对单个 exam 文件生成 scenario_count 个问题场景输出。
    每个场景内最多尝试 max_retries 次以保证数据完整性。
    若任一场景无法在重试次数内获得完整数据，
    则该 exam 文件全部不写入，以保证数据一致性。
    """
    exam_content = read_txt_file(file_path)
    scenarios = []
    
    # 针对每个 exam 文件生成 scenario_count 个场景
    for i in range(scenario_count):
        scenario_valid = None
        for attempt in range(max_retries):
            print(f"处理文件 {file_path}，生成第 {i+1} 个场景，第 {attempt+1} 次尝试...")
            resp = llm_processing(exam_content, model_name, client)
            if hasattr(resp.choices[0].message, "tool_calls"):
                for tool_call in resp.choices[0].message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        print(f"Error parsing JSON: {str(e)}")
                        continue
                    if validate_arguments(arguments):
                        scenario_valid = arguments
                        break
            if scenario_valid is not None:
                break
            else:
                time.sleep(10)
        
        if scenario_valid is None:
            print(f"文件 {file_path} 的第 {i+1} 个输出未生成完整有效数据，整个文件跳过写入。")
            return  # 如果任一场景失败，则整个文件不写入，保证数据数量一致性
        scenarios.append(scenario_valid)
    
    # 所有场景均通过校验后，一次性写入所有输出
    for scenario in scenarios:
        write_full_answer(scenario, output_folder)
        write_csv(scenario, output_folder, os.path.basename(file_path))
    print(f"文件 {file_path} 的全部 {scenario_count} 个输出已完整写入")


def exam_main(config):
    input_folder = config.get("exam_input_folder")
    output_folder = config.get("exam_output_folder")
    retries = config.get('retries', 3)
    scenario_count = config.get('exam_scenarios_count', 8)
    model_name = config.get('model_name', 'gpt-4o')
    openai_api_key = config.get('OPENAI_API_KEY')
    base_url = config.get('OPENAI_BASE_URL')
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=base_url,
    )
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        process_exam_file(file_path, output_folder, model_name, client, scenario_count=scenario_count, max_retries=retries)
        time.sleep(10)