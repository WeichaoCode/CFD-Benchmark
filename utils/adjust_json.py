import json

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/data/cfd_prompts.json"  # Input and output are the same file


def add_key_to_json(filename, key="key"):
    # Load the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # Add "key" to each prompt
    for index, problem in enumerate(data["prompts"], start=1):
        problem[key] = index  # Assign a unique key

    # Save the updated JSON back to the same file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Updated JSON saved in {filename}")


def add_instruction_to_json(filename, instruction):
    # Load the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # Append instruction to each prompt
    # instruction = ("Write a Python script to solve this problem and return only the complete Python code, without any "
    #                "explanation or additional text.")

    for problem in data["prompts"]:
        problem["prompt"] += instruction  # Append instruction to the existing prompt

    # Save the updated JSON back to the same file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Updated JSON saved in {filename}")


def change_problems_to_json(filename, changed_key, changed_value):
    # Load the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    for problem in data["problems"]:
        problem[changed_key] = changed_value  # Append instruction to the existing prompt

    # Save the updated JSON back to the same file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Updated JSON saved in {filename}")


# change_problems_to_json("/opt/CFD-Benchmark/data/cfd_problems.json", "time",
#                         "Final time T = 2, start time is 0")

add_key_to_json("/opt/CFD-Benchmark/data/cfd_prompts.json", "key")
