import json
import os
import time
import csv
import base64
import cv2  # OpenCV 库
import numpy as np
from openai import OpenAI



def is_video_file(file_path):
    """检查是否为视频文件（基于扩展名）"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    return os.path.splitext(file_path)[-1].lower() in video_extensions


def extract_frames_from_video(video_path, interval_seconds=5):
    """从视频中按时间间隔提取帧，返回 Base64 编码的帧列表"""
    frames_base64 = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Skipping unreadable video file: {video_path}")
        return None  # 跳过无法打开的视频
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds) if fps > 0 else 1
    frame_count = 0

    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            retval, buffer = cv2.imencode('.jpg', frame)
            if retval:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(frame_base64)
        frame_count += 1
        success, frame = cap.read()
    
    cap.release()
    
    return frames_base64 if frames_base64 else None  # 确保不传递空帧列表


def generate_visual_content_description(frames_base64, client, model_name):
    if frames_base64 is None:
        return None  # 若无有效帧直接返回 None

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following video frames are extracted from the video at fixed time intervals (not consecutive frames). Please describe the scene, character actions, environmental changes, and other important details in the video."
                }
            ]
        }
    ]

    # 附加每帧的 Base64 图片数据（伪造 image_url 结构）
    for frame in frames_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
        })

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=300,
    )

    return response.choices[0].message.content


def save_video_description(file_name, description, save_folder="../results/video_descriptions_v1"):
    """保存视频描述至文本文件"""
    if description is None:
        print(f"Skipping description save for {file_name} (No valid frames extracted).")
        return
    
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, f"{file_name}_description.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(description)
    print(f"Video description saved to: {file_path}")


def validate_arguments(arguments):
    """
    校验返回的 arguments 是否包含 IRAC 框架的所有必填组件及字段，
    其中 IRAC 组件包括：issue_spotting, rule_recall, rule_application, rule_conclusion
    """
    required_components = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]
    required_fields = [
        "m1_question", "m1_option_A", "m1_option_B", 
        "m1_option_C", "m1_option_D", "m1_answer", "m1_explanation"
    ]
    for comp in required_components:
        if comp not in arguments:
            print(f"Error: 缺少必备组件 {comp}")
            return False
        for field in required_fields:
            if field not in arguments[comp]:
                print(f"Error: 在组件 {comp} 中缺少字段 {field}")
                return False
    return True


def llm_processing_video(content, model_name, client):
    """调用 GPT-4o 生成基于视频描述的 IRAC 框架问题（一个场景）"""
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

I have a video about HSE. The description of this video is: {content}. Note that the specific region applicable to this scenario should be determined from the content.

Based on this video description, generate scenario-based questions that adhere strictly to the IRAC framework. For the scenario, generate questions addressing four key components:

1. Issue Spotting – Identify the critical safety or legal issue presented in the scenario.
2. Rule Recall – State the relevant legal rule(s) or regulation(s) applicable to the identified issue.
3. Rule Application – Explain in detail how the legal rule applies to the given facts. The explanation must include:
   - Specific inferences drawn from the facts.
   - A detailed reasoning process that connects the facts to the rule.
   - A clear statement of the expected outcome based on these inferences.
4. Rule Conclusion – Present the final legal conclusion derived from the rule application.

For each of these four IRAC components, generate a multiple choice question with only one correct answer (labeled as m1_question).

For every question, provide:
- A correct answer (m1_answer).
- A comprehensive explanation (m1_explanation) that meets the following criteria:
   • Correctness: The explanation must be factually accurate with no misstatements.
   • Analysis: It must include detailed inferences and reasoning.
   • Specificity: It must reference a specific, up-to-date HSE regulation (with name and clause number if applicable).
   • Uniqueness: It must be unique and tailored to the scenario.

Additionally, include distractor options (e.g. trap option m1_trap_answer) with subtle errors. In the explanations, detail why the correct answer is valid and identify the misleading elements in distractors.

Generate the complete set of questions, answers, and explanations for each IRAC component based on the video description.
                    """
                }
            ]
        }
    ]
    
    irac_property = {
        "type": "object",
        "properties": {
            "m1_question": {"type": "string", "description": "Multiple choice question body with one correct answer"},
            "m1_option_A": {"type": "string", "description": "Option A for m1_question"},
            "m1_option_B": {"type": "string", "description": "Option B for m1_question"},
            "m1_option_C": {"type": "string", "description": "Option C for m1_question"},
            "m1_option_D": {"type": "string", "description": "Option D for m1_question"},
            "m1_answer": {"type": "string", "description": "Answer for m1_question", "enum": ["A", "B", "C", "D"]},
            "m1_explanation": {"type": "string", "description": "Explanation for m1_question"},
        },
        "required": [
            "m1_question", "m1_option_A", "m1_option_B", "m1_option_C", "m1_option_D",
            "m1_answer", "m1_explanation"
        ],
    }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "regulation_questions",
            "description": "Ask questions about the provided regulation based on the IRAC framework.",
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


def write_csv(arguments, output_folder):
    """
    将 IRAC 框架问题写入 CSV 文件，校验数据完整性后再写入。
    如果写入过程中发生异常，则返回 False（方便上层重试）。
    """
    scenario = arguments.get("scenario", "No scenario provided")
    irac_components = ["issue_spotting", "rule_recall", "rule_application", "rule_conclusion"]

    # 检查各组件字段是否完整
    for component in irac_components:
        required_fields = [f"m1_question", f"m1_option_A", f"m1_option_B",
                           f"m1_option_C", f"m1_option_D", f"m1_answer", f"m1_explanation"]
        for field in required_fields:
            if field not in arguments.get(component, {}):
                print(f"Warning: 在组件 {component} 中缺少字段 {field}")
                return False
    
    try:
        for component in irac_components:
            data = arguments[component]
            mode = "m1"
            question = data[f"{mode}_question"]
            file_name = os.path.join(output_folder, f"{component}/m1.csv")

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow((
                    f"{scenario} {question}",
                    data[f"{mode}_option_A"],
                    data[f"{mode}_option_B"],
                    data[f"{mode}_option_C"],
                    data[f"{mode}_option_D"],
                    data[f"{mode}_answer"],
                    data[f"{mode}_explanation"]
                ))
        return True
    except Exception as e:
        print(f"Error writing CSV files: {str(e)}")
        return False


def write_full_answer(arguments, output_folder):
    """将完整响应以 TXT 格式写入到文件中"""
    output_path = os.path.join(output_folder, "full_output.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f"--- Response at {timestamp} ---\n")
        f.write(json.dumps(arguments, ensure_ascii=False, indent=2))
        f.write("\n\n")


def get_shortened_filename(video_file):
    """获取不带扩展名的原始文件名，过长则截断"""
    base_name = video_file.rsplit('.', 1)[0].replace('*', ' ').strip()
    max_length = 30
    if len(base_name) > max_length:
        import re
        match = re.search(r'(\d+)$', base_name)
        if match:
            seq = match.group(1)
            prefix_length = max_length - len(seq) - 3
            return base_name[:prefix_length] + "..." + seq
        else:
            return base_name[:max_length-3] + "..."
    return base_name


def process_video_file(video_file, video_input_folder, output_folder, model_name, client, file_retry_max=3, csv_retry_max=3):
    """
    对单个视频文件进行处理：
      - 提取帧并生成视频描述
      - 调用 GPT-4o 获取 IRAC 问题（仅一个场景）
      - 重试 CSV 写入（内部 csv_retry_max 次）
      - 如果整个流程在 file_retry_max 内成功，则保存描述并标记该视频为处理成功
    """
    video_path = os.path.join(video_input_folder, video_file)
    file_name = get_shortened_filename(video_file)
    attempt = 0
    while attempt < file_retry_max:
        print(f"Processing {video_file} (attempt {attempt+1})...")
        # 提取视频帧
        frames = extract_frames_from_video(video_path)
        if frames is None:
            print(f"Skipping {video_file} due to no extractable frames.")
            return False
        
        # 生成视频描述
        visual_desc = generate_visual_content_description(frames, client, model_name)
        if visual_desc is None:
            print(f"Skipping description generation for {video_file}.")
            return False
        
        # 调用 GPT-4o 获取 IRAC 问题
        resp = llm_processing_video(visual_desc, model_name, client)
        # 假定 GPT 返回的 response 包含 tool_calls 数组
        response_valid = False
        for tool_call in resp.choices[0].message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception as e:
                print(f"Error parsing GPT response for {video_file}: {str(e)}")
                continue
            # 可选：对返回数据进行完整性校验
            if not validate_arguments(arguments):
                print(f"Validation failed for {video_file}.")
                continue
            
            # 写入完整响应（无重试）
            write_full_answer(arguments, output_folder)
            
            # CSV 写入重试
            csv_attempt = 0
            csv_success = False
            while csv_attempt < csv_retry_max:
                csv_success = write_csv(arguments, output_folder)
                if csv_success:
                    break
                csv_attempt += 1
                print(f"CSV writing attempt {csv_attempt} failed for {video_file}, retrying...")
                time.sleep(10)
            
            if csv_success:
                # 若成功，则更新 scenario 字段（可由 GPT 返回的数据中选择一个合适的描述）
                # 这里假设 GPT 返回的数据中已经有 "scenario" 字段
                # 如果没有则可以用 visual_desc 代替
                if "scenario" not in arguments:
                    arguments["scenario"] = visual_desc
                response_valid = True
                break
        if response_valid:
            # 若处理成功，保存视频描述并返回 True
            save_video_description(file_name, visual_desc, save_folder="../results/video_descriptions_v1")
            return True
        attempt += 1
        print(f"Retrying full processing for {video_file} (attempt {attempt+1}) after delay...")
        time.sleep(10)
    return False

def video_main(config):
    video_input_folder = config.get('video_input_folder', "")
    output_folder = config.get('video_output_folder', "")
    retries = config.get('retries', 3)

    model_name = config.get('model_name', 'gpt-4o')
    openai_api_key = config.get('OPENAI_API_KEY')
    base_url = config.get('OPENAI_BASE_URL')
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=base_url,
    )

    for video_file in os.listdir(video_input_folder):
        success = process_video_file(video_file, video_input_folder, output_folder, model_name, client, file_retry_max=retries, csv_retry_max=retries)
        if success:
            print(f"Successfully processed {video_file}")
        else:
            print(f"Failed to process {video_file} after max attempts.")

        # 视频间暂停
        time.sleep(10)

