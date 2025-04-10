from utils import PromptGenerator, LLMCodeGenerator, SolverPostProcessor
import os

# STEP 1: generate the prompts
# Set the root directory (you can also use os.getcwd() if running from root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(ROOT_DIR, 'PDE_Benchmark')  # PDE_Benchmark root

# Initialize the PromptGenerator
generator_prompt = PromptGenerator(root_dir)

# Load the problem data from the input JSON
generator_prompt.load_problem_data()

# Create prompts from the data
prompts = generator_prompt.create_prompts()

# Save to prompts.json (will skip if already exists)
generator_prompt.save_prompts(prompts)

# STEP 2: call LLM API to get generated code
# Set the model and prompt JSON filename
llm_model = "o3-mini"  # "gpt-4o", ""o3-mini, "sonnet-3.5", "haiku", "lama 4", "gemma 3"
prompt_json = "prompts.json"  # the file under ./prompt/

# # Instantiate the class
# generator_llm = LLMCodeGenerator(llm_model, prompt_json)
#
# # Call the API to generate code for each task
# generator_llm.call_api()

# STEP 3: post-process the generate code and compare the loss and images
# Create the post-processor
processor = SolverPostProcessor(llm_model, prompt_json)

# Run the full post-processing pipeline
processor.run_all()
