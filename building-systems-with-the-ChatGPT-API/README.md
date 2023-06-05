# Building Systems with the ChatGPT API

- https://learn.deeplearning.ai/chatgpt-building-system/lesson/1/introduction
- https://twitter.com/AndrewYNg/status/1663984377918001153

Notes:

## L1 Language Models, the Chat Format and Tokens

Very basic stuff.

## L2 Evaluate Inputs - Classification

Classification with LLM feels very magical, you literally just describe the categories and ask what it is, like this code snippet:

```py
delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category.
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
...

"""
user_message = f"""\
I want you to delete my profile and all of my user data"""
messages =  [
{'role':'system',
 'content': system_message},
{'role':'user',
 'content': f"{delimiter}{user_message}{delimiter}"},
]
response = get_completion_from_messages(messages)

"""
{
  "primary": "Account Management",
  "secondary": "Close account"
}
"""
```

## L3 Evaluate Inputs - Moderation

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- How to deal with potential prompt injections

## L4 Process Inputs - Chain of Thought Reasoning

Described in [[2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) and [[2205.11916] Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), guiding LLM to think step-by-step could greatly improve the performance.

This lesson's CoT prompt is one step further, which asks the LLM to say its reasoning, and extracts out the final output after the step by step reasoning.

The example:

```py
delimiter = "####"
system_message = f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags,\
i.e. {delimiter}.

Step 1:{delimiter} First decide whether the user is \
asking a question about a specific product or products. \
Product cateogry doesn't count.

Step 2:{delimiter} If the user is asking about \
specific products, identify whether \
the products are in the following list.
All available products:
1. Product: TechPro Ultrabook
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-UB100
   Warranty: 1 year
   Rating: 4.5
   Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5 processor
   Description: A sleek and lightweight ultrabook for everyday use.
   Price: $799.99

2. ...

Step 3:{delimiter} If the message contains products \
in the list above, list any assumptions that the \
user is making in their \
message e.g. that Laptop X is bigger than \
Laptop Y, or that Laptop Z has a 2 year warranty.

Step 4:{delimiter}: If the user made any assumptions, \
figure out whether the assumption is true based on your \
product information.

Step 5:{delimiter}: First, politely correct the \
customer's incorrect assumptions if applicable. \
Only mention or reference products in the list of \
5 available products, as these are the only 5 \
products that the store sells. \
Answer the customer in a friendly tone.

Use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Step 4:{delimiter} <step 4 reasoning>
Response to user:{delimiter} <response to customer>

Make sure to include {delimiter} to separate every step.
"""

user_message = f"""
by how much is the BlueWave Chromebook more expensive \
than the TechPro Desktop"""

messages =  [
{'role':'system',
 'content': system_message},
{'role':'user',
 'content': f"{delimiter}{user_message}{delimiter}"},
]

response = get_completion_from_messages(messages)
print(response)

"""
Step 1:#### The user is asking a question about two specific products, the BlueWave Chromebook and the TechPro Desktop.
Step 2:#### The prices of the two products are as follows:
- BlueWave Chromebook: $249.99
- TechPro Desktop: $999.99
Step 3:#### The user is assuming that the BlueWave Chromebook is more expensive than the TechPro Desktop.
Step 4:#### The assumption is incorrect. The TechPro Desktop is actually more expensive than the BlueWave Chromebook.
Response to user:#### The BlueWave Chromebook is actually less expensive than the TechPro Desktop. The BlueWave Chromebook is priced at $249.99, while the TechPro Desktop is priced at $999.99.
"""

try:
    final_response = response.split(delimiter)[-1].strip()
except Exception as e:
    final_response = "Sorry, I'm having trouble right now, please try asking another question."

print(final_response)

"""
I'm sorry, but we do not sell TVs at this time. Our store specializes in computers and laptops. However, if you are interested in purchasing a computer or laptop, please let me know and I would be happy to assist you.
"""
```

## L5 Process Inputs - Chaining Prompts

- More focused: breaks down a complex task
- Works around context limitations: max tokens for input prompt and output responses
- Reduced costs: per per token

Think about the model as a reasoning engine, give it the information it needs.

## L6 Check outputs

Check output for potentially harmful content

```py
moderation_output = openai.Moderation.create(
    input=final_response_to_customer
)
"""
{
  "categories": {
    "hate": false,
    "hate/threatening": false,
    "self-harm": false,
    "sexual": false,
    "sexual/minors": false,
    "violence": false,
    "violence/graphic": false
  },
  "category_scores": {
    "hate": 4.2486033e-07,
    "hate/threatening": 5.676476e-10,
    "self-harm": 2.9144967e-10,
    "sexual": 2.243237e-06,
    "sexual/minors": 1.2526144e-08,
    "violence": 5.949349e-06,
    "violence/graphic": 4.4063694e-07
  },
  "flagged": false
}
"""
```

Check if output is factually based on the provided product information. The prompt looks like this:

````py
system_message = f"""
You are an assistant that evaluates whether \
customer service agent responses sufficiently \
answer customer questions, and also validates that \
all the facts the assistant cites from the product \
information are correct.
The product information and user and customer \
service agent messages will be delimited by \
3 backticks, i.e. ```.
Respond with a Y or N character, with no punctuation:
Y - if the output sufficiently answers the question \
AND the response correctly uses product information
N - otherwise

Output a single letter only.
"""

customer_message = f"""tell me about the ..."""

product_information = """{ ... }"""

q_a_pair = f"""
Customer message: ```{customer_message}```
Product information: ```{product_information}```
Agent response: ```{final_response_to_customer}```

Does the response use the retrieved information correctly?
Does the response sufficiently answer the question

Output Y or N
"""

messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': q_a_pair}
]

response = get_completion_from_messages(messages, max_tokens=1)
print(response)
````

## L7 Evaluation - Build an End-to-End System

Process of building a LLM application

1. Tune prompts on handful of examples
2. Add additional "tricky" examples opportunistically
3. Develop metrics to measure performance on examples
4. Collect randomly sampled set of examples to tune to (development set hold-out cross validation set)
   1. (More important if accuracy is critical to your app)
5. Collect and use a hold-out test set

## L8 Evaluation part I - single right answer

Evaluate LLM responses when there is a single "right answer".

Recap the steps in L7, then add regression testing on previous test cases, evaluate test cases by comparing to the ideal answers. Example of test cases:

```py
msg_ideal_pairs_set = [

    # eg 0
    {'customer_msg':"""Which TV can I buy if I'm on a budget?""",
     'ideal_answer':{
        'Televisions and Home Theater Systems':set(
            ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']
        )}
    },

    # ...
]
```

## L9 Evaluation Part II - multiple right answers

Evaluate LLM responses where there isn't a single "right answer."

Check if the LLM's response agrees with or disagrees with the ideal/expert answer (human generated)

- This evaluation prompt is from the [OpenAI evals](https://github.com/openai/evals/blob/main/evals/registry/modelgraded/fact.yaml) project.
- [BLEU score](https://en.wikipedia.org/wiki/BLEU): another way to evaluate whether two pieces of text are similar or not.
