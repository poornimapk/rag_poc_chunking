from llama_index.llms.ollama import Ollama

llm = Ollama(model="mistral")

response = llm.complete("Why are the plants green?")

print(response)

