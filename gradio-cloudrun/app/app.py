""" 
    Gradio App that calls Vertex AI prediction endpoint
"""

import gradio as gr

from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/3494063235118661632"  # <---- CHANGE THIS !!!!
)

examples = [
  ["Please answer to the following question. Who is going to be the next Ballon d'or?"],
  ["Q: Can Barack Obama have a conversation with George Washington? Give the rationale before answering."],
  ["Summarize the following text: Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital. Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well. Therefore, Peter stayed with her at the hospital for 3 days without leaving."],
  ["Please answer the following question: What is the boiling point of water?"],
  ["Answer the following question by detailing your reasoning: Are Pokemons alive?"],
  ["Translate to German: How old are you?"],
  ["Generate a cooking recipe to make bolognese pasta:"],
  ["Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"],
  ["Premise:  At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?"],
  ["Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"],
  ["""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?"""]
]

title = "LLM demo: Flan-T5 XXL in Vertex AI Prediction"
description = "This demo shows [Flan-T5-XX-large](https://huggingface.co/google/flan-t5-xxl), a 11B model deployed in Vertex AI Prediction. Note that T5 expects a very specific format of the prompts, so the examples below are not necessarily the best prompts to compare."
article = "**INSTRUCTIONS. PLEASE READ:** \n \
  * REFRESH the page in case of errors or after some idle time. \n \
  * This version does not contain any fine-tuning, it's just the checkpoints as published in [the paper](https://arxiv.org/abs/2210.11416.pdf). \n \
  * Do NOT try to input code. This version does NOT work well with code. \n \
  * Although works with other languages (Spanish), try mainly English examples. \n \
  * Resources for this demo (CPU, GPU, RAM) costs **USD 170 per day**, so may be offline specially outside of CET time \n\n\n \
  Contact: rafaelsanchez@google.com. If you find any good examples, please let me know and will add them to the list "


def inference(text):
  response = endpoint.predict([[str(text)]])
  return str(response.predictions[0][0])


io = gr.Interface(
  inference,
  gr.Textbox(lines=3),
  outputs=[
    gr.Textbox(lines=3, label="Flan T5")
  ],
  title=title,
  description=description,
  examples=examples,
  article=article,
  thumbnail='./googlecloud.png'
)

io.launch(server_name="0.0.0.0", server_port=7860)