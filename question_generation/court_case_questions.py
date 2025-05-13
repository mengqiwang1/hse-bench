import json
from openai import OpenAI
import csv
from regulation_questions import write_csv

def llm_processing_court_case(region_name, original_text, model_name, client):
    with open("../input_files/IRAC_Framework.txt", mode='rb') as file:
        irac_rules = file.read()
    messages = [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"""
I have a court case about HSE in {region_name}. I now need to apply the IRAC framework to generate test questions based on the court case. 

First, generalise the court case to a similar scenario that clearly states who do what at where. 
Second, we need four types of questions based on the IRAC framework, including issue spotting, rule recall, rule application, and rule conclusion.
We require the option of issue spotting, which is the issue to be claimed in this scenario. The rule recall is the rule that is relevant to the issue. The rule application is the application of the rule to the scenario. The rule conclusion is the conclusion of the scenario based on the rule application.

Third, we need three modes of questions for each question type: 1. Single choice(m1_question), 2. Multiple choice(m2_question), 3. True/False(tf_question).

The details about this IRAC framework can be found here: {irac_rules}

The context of the court case is: {original_text}

              """
            }
          ]
        }
    ]

    irac_property = {
        "type": "object",
        "properties": {
            "tf_question": {"type": "string", "description": "True or False question body"},
            "tf_answer": {"type": "string", "description": "Answer for the True or False question", "enum": ["T", "F"]},
            "m1_question": {"type": "string", "description": " multiple choice question body with one correct answer"},
            "m1_option_A": {"type": "string", "description": " option A for m1_question"},
            "m1_option_B": {"type": "string", "description": " option B for m1_question"},
            "m1_option_C": {"type": "string", "description": " option C for m1_question"},
            "m1_option_D": {"type": "string", "description": " option D for m1_question"},
            "m1_answer": {"type": "string", "description": " answer for m1_question", "enum": ["A", "B", "C", "D"]},
            "m2_question": {"type": "string", "description": " multiple choice question body with one or more correct answers"},
            "m2_option_A": {"type": "string", "description": " option A for m2_question"},
            "m2_option_B": {"type": "string", "description": " option B for m2_question"},
            "m2_option_C": {"type": "string", "description": " option C for m2_question"},
            "m2_option_D": {"type": "string", "description": " option D for m2_question"},
            "m2_option_E": {"type": "string", "description": " option E for m2_question"},
            "m2_answer": {
                "type": "string",
                "description": " answer for m2_question",
                "enum": [
                    "A", "B", "C", "D", "E",
                    "A,B", "A,C", "A,D", "A,E",
                    "B,C", "B,D", "B,E",
                    "C,D", "C,E",
                    "D,E",
                    "A,B,C", "A,B,D", "A,B,E",
                    "A,C,D", "A,C,E",
                    "A,D,E",
                    "B,C,D", "B,C,E",
                    "B,D,E",
                    "C,D,E",
                    "A,B,C,D", "A,B,C,E",
                    "A,B,D,E", "A,C,D,E",
                    "B,C,D,E",
                    "A,B,C,D,E"
                ]
            },
        },
        "required": ["tf_question", "tf_answer", "m1_question", "m1_option_A", "m1_option_B", "m1_option_C", "m1_option_D", "m1_answer", "m2_question", "m2_option_A", "m2_option_B", "m2_option_C", "m2_option_D", "m2_option_E", "m2_answer"],
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "court_case_questions",
            "description": "Ask questions about the provided court case in different types and modes. ",
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


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# Main function for processing court cases
def court_case_main(config):
    court_cases = config.get('court_case', [])
    output_folder = config.get('court_case_output_folder', '')
    model_name = config.get('model_name', 'gpt-4o')
    openai_api_key = config.get('OPENAI_API_KEY')
    base_url = config.get('OPENAI_BASE_URL')
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=base_url,
    )

    for court_case in court_cases:
        court_case_name = court_case.get('name')
        region_name = court_case.get('region_name')
        court_case_path = court_case.get('input_file_path')

        with open(court_case_path, mode='rb') as file:
            file_content = file.read()

        resp = llm_processing_court_case(region_name, file_content, model_name, client)

        for tool_call in resp.choices[0].message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            write_csv(court_case_name, arguments, output_folder)

