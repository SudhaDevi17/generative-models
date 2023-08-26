from dotenv import load_dotenv
import os
load_dotenv()

from langchain import PromptTemplate
import numpy as np
# # To interact with LLMs in LangChain we need the following modules loaded
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import LLMChain
from better_profanity import profanity

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('hugging_face_secret')
os.environ["SERPAPI_API_KEY"] = os.getenv('serpapi_secret')


# Our template for Jekyll will instruct it on how it should respond, and what variables (using the {text} syntax) it should use.
jekyll_template = """
You are a social media post commenter, you will respond to the following post with a {sentiment} response. 
Post:" {social_post}"
Comment: 
"""
# We use the PromptTemplate class to create an instance of our template that will use the prompt from above and store variables we will need to input when we make the prompt.
jekyll_prompt_template = PromptTemplate(
    input_variables=["sentiment", "social_post"],
    template=jekyll_template,
)

# Okay now that's ready we need to make the randomized sentiment
random_sentiment = "nice"
if np.random.rand() < 0.3:
    random_sentiment = "mean"
# We'll also need our social media post:
social_post = "I can't believe I'm learning about LangChain in this MOOC, there is so much to learn and so far the instructors have been so helpful. I'm having a lot of fun learning! #AI #Databricks"

# Let's create the prompt and print it out, this will be given to the LLM.
jekyll_prompt = jekyll_prompt_template.format(
    sentiment=random_sentiment, social_post=social_post
)
print(f"Jekyll prompt:{jekyll_prompt}")

model_id = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="../working/cache")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="../working/cache")
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device_map='auto'
)
jekyll_llm = HuggingFacePipeline(pipeline=pipe)

jekyll_chain = LLMChain(
    llm=jekyll_llm,
    prompt=jekyll_prompt_template,
    output_key="jekyll_said",
    verbose=False,
)  # Now that we've chained the LLM and prompt, the output of the formatted prompt will pass directly to the LLM.

# To run our chain we use the .run() command and input our variables as a dict
jekyll_said = jekyll_chain.run(
    {"sentiment": random_sentiment, "social_post": social_post}
)

# Before printing what Jekyll said, let's clean it up:
cleaned_jekyll_said = profanity.censor(jekyll_said)
print(f"Jekyll said:{cleaned_jekyll_said}")
