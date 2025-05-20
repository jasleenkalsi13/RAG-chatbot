from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os
import streamlit

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['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']

def main():
    # Get embedding for a word.
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
