import os
from openai import OpenAI
import csv
import json



def llm_processing(region_name, regulation_name, original_text, model_name, client):
    with open("../input_files/IRAC_Framework.txt", mode='rb') as file:
        irac_rules = file.read()

    messages = [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"""
The details about this IRAC framework can be found here: {irac_rules}. I now need to apply this framework to the HSE field in a way that challenges examinees to perform multi-layered legal analysis.

I have a regulation about HSE in location: {region_name}, titled {regulation_name}. The context of the regulation is: {original_text}

Based on these regulation contexts, generate scenario-based questions in {region_name} that adhere strictly to the IRAC framework. For each scenario, generate questions addressing four key components:

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

Make sure to include the location in the scenario. 

Additionally, ensure adversarial generation of answer options by including distractor options:
   - For multiple-choice questions, in addition to the correct answer (m1_answer), also give a trap option (m1_trap_answer) that is roughly half are correct (or partially correct) and half are incorrect. It should seems to be correct for non-expert examinees or LLMs, which correlates to some contexts in the scenario.
   - The distractor options should be plausible, referencing real-world regulatory elements but include subtle errors (e.g., incorrect clause numbers, misinterpretations of the rule, or irrelevant legal principles) to challenge examinees.
In summary, the generated questions should compel the examinee to:
   • Identify nuanced safety issues.
   • Recall and articulate specific legal rules.
   • Apply these rules with detailed legal analysis and inference.
   • Reach a reasoned legal conclusion supported by specific regulatory references.
   • Critically evaluate adversarial answer options.

Generate the complete set of questions, answers for each IRAC component and each question format.

              """
            }
          ]
        }
    ]

    irac_property = {
        "type": "object",
        "properties": {
            "m1_question": {"type": "string", "description": " multiple choice question body with one correct answer"},
            "m1_option_A": {"type": "string", "description": " option A for m1_question"},
            "m1_option_B": {"type": "string", "description": " option B for m1_question"},
            "m1_option_C": {"type": "string", "description": " option C for m1_question"},
            "m1_option_D": {"type": "string", "description": " option D for m1_question"},
            "m1_answer": {"type": "string", "description": " answer for m1_question", "enum": ["A", "B", "C", "D"]},
        },
        "required": ["m1_question", "m1_option_A", "m1_option_B", "m1_option_C", "m1_option_D", "m1_answer"],
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "regulation_questions",
            "description": "Ask questions about the provided regulation in different types and modes. ",
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



def write_csv(reg_name, arguments, output_folder):
    scenario = arguments["scenario"]
    for question_type in ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]:
        argument = arguments[question_type]
        question = argument[f"m1_question"]
        file_name = os.path.join(output_folder, f"{question_type}/m1.csv")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow((f"{scenario} {question}", argument[f"m1_option_A"], argument[f"m1_option_B"], argument[f"m1_option_C"], argument[f"m1_option_D"], argument[f"m1_answer"], reg_name))


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def split_string_into_parts(content, num_parts=10):
    """将字符串平均分割成指定份数"""
    part_length = len(content) // num_parts  # 每份的长度
    parts = [content[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
    return parts


def regulation_main(config):
    regulations = config.get('regulation', [])
    output_folder = config.get('regulation_output_folder', '')
    retries = config.get('retries', 3)
    split_size = config.get('regulation_split_size', 4)
    model_name = config.get('model_name', 'gpt-4o')
    openai_api_key = config.get('OPENAI_API_KEY')
    base_url = config.get('OPENAI_BASE_URL')
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=base_url,
    )

    # Iterate over the list of regulations
    for regulation in regulations:
        regulation_name = regulation.get('name')
        region_name = regulation.get('region_name')
        regulation_path = regulation.get('input_file_path')

        # Read the regulation file, split into parts for processing and generating questions
        file_content = read_txt_file(regulation_path)
        reg_parts = split_string_into_parts(file_content, split_size)

        # Iterate over the parts of the regulation document
        for index, reg_part in enumerate(reg_parts):
            print("-" * 40)
            print(f"Part {index}:")

            i = 0
            while i < retries:
                try:
                    resp = llm_processing(region_name, regulation_name, reg_part, model_name, client)
                    for tool_call in resp.choices[0].message.tool_calls:
                        arguments = json.loads(tool_call.function.arguments)
                        write_csv(regulation_name, arguments, output_folder)
                    i = retries  # Exit retry loop after success
                except Exception as e:
                    print(f"Error processing part {index}, error: {e}, retrying... ({i+1}/{retries})")
                    i += 1
