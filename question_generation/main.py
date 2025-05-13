import json

from court_case_questions import court_case_main
from rephrase_questions import rephrase_main
from regulation_questions import regulation_main
from safety_exam_questions import exam_main
from video_questions import video_main


# Function to load configuration from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)




if __name__ == "__main__":
    config = load_config("config.json")
    regulation_main(config)
    court_case_main(config)
    exam_main(config)
    video_main(config)
    rephrase_main(config)