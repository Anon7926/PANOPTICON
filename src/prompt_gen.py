from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.PanoHandler import Panorama, PanoPlus
from utils.pii_handler import sample_pii, match_pii
import torch
from random import sample, choice
from jinja2 import Template
from datasets import Dataset
import argparse as ap
import math

parser = ap.ArgumentParser()
parser.add_argument("--num-chunks", type=int, required=True)
parser.add_argument("--chunk-index", type=int, required=True)
parser.add_argument('--model-name', default="meta-llama/Llama-3.1-8B-Instruct", type=str, required=False)
args = parser.parse_args()

entries = []
#Variation in scenarios for prompt generation.
CATEGORIES = {
    "Professional": [ #3 Scenarios
        "Resume, CV or Career Materials",
        "HR and Internal communications",
        "Advertise Professional Services",
    ], 

    "Finance": [ #3 Scenarios
        "Budget planning",
        "Tax bracket/deduction inquiry",
        "Mortgage or loan application",
    ],

    "Health": [ #3 Scenarios
        "Medical Provider Communications",
        "Medical history summary",
        "Insurance, Billing, or Administrative Assistance",
    ],

    "Relationship": [ #3 Scenarios
        "Message to partner/family member",
        "Relationship Advice",
        "Family trip planning",
    ],

    "Government": [ #3 Scenarios
        "Document renewal or recovery assistance",
        "Social services inquiry",
        "Identity verification request"
    ],

    "Social": [ #3 Scenarios
        "Social media post",
        "Email or Direct Message assistance",
        "Online Marketplace Assistance"
    ],

    "Benign": [ #20 Scenarios
        "Productivity tips",
        "Travel recommendations",
        "Book/movie suggestions",
        "General knowledge questions",
        "Cooking recipes",
        "Gardening advice",
        "Fitness routines",
        "Language learning assistance",
        "Creative writing prompts",
        "DIY project ideas",
        "Meditation and mindfulness techniques",
        "Time management strategies",
        "Hobby exploration suggestions",
        "Study tips and techniques",
        "Home organization hacks", 
        "Stress management methods",
        "Pet care advice",
        "Outdoor activity recommendations",
        "Video game suggestions",
        "Music or podcast recommendations",
        ]
}


        
MODEL = args.model_name #Change this to a particular model if you want to omit the argument but use a different model

#Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'gen_prompts.md'
benign_system_prompt = current_directory.parent / 'prompts' / 'benign_prompt.md'
#response_system_prompt = current_directory.parent / 'prompts' / 'response.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction_md = f.read()

with open(benign_system_prompt, 'r', encoding='utf-8') as b:
    benign_instruction_md = b.read()

'''with open(response_system_prompt, 'r', encoding='utf-8') as r:
    response_md = r.read()''' 

template = Template(instruction_md)
benign_template = Template(benign_instruction_md)

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=.4, top_p=0.9)


#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs

num_chunks = args.num_chunks
chunk_index = args.chunk_index

n = len(ids)
chunk_size = math.ceil(n / num_chunks)

start = chunk_index * chunk_size
end = min(start + chunk_size, n)

def gen_content(uid):
    data =  []
    content = []

    article = pano.articles.filter(lambda x:x['id'] == uid)
    article = article['text'][0] #Ensuring no duplicate articles

    social = pano.socials.filter(lambda x:x['id'] == uid)
    forums = pano.forums.filter(lambda x:x['id'] == uid)
    reviews = pano.reviews.filter(lambda x:x['id'] == uid)
    comments = pano.comments.filter(lambda x:x['id'] == uid)
    ads = pano.ads.filter(lambda x:x['id'] == uid)

    data.append(social['text'])
    data.append(forums['text'])
    data.append(reviews['text'])
    data.append(comments['text'])
    data.append(ads['text'])

    for d in data:
        for post in d:
            content.append(post)
    #Getting 4 or fewer random sample online writings and article for tone guidance and context
    return sample(content, min(len(content), 4)), article

#Main loop
for uid in ids[start:end]: 

    content, article = gen_content(uid)

    #Getting relevat PII
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0] #Ensuring duplicate PII is not passed

    pool = 0
    for category in CATEGORIES.keys():
        if category == 'Benign':
            scenario = choice(CATEGORIES[category])
            instruction = benign_template.render(scenario=scenario)
        else:
            pii = sample_pii(pool, full_pii) 
            scenario = choice(CATEGORIES[category]) #Selecting a scenario from list of categories
            instruction = template.render(category=category, scenario=scenario, pii=pii)
            pool+=1 #Next category



        prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article} Content: {content}.'}]

        #Generate prompt
        prompt = generator(prompts, truncation=True, max_new_tokens=120)
        prompt = prompt[0]['generated_text']

        #Generate LLM output
        #responses = [{'role':'system', 'content':f'{response_md}'}, {'role':'user', 'content':f'{prompt}'}]
        #response = generator(responses, truncation=True, max_new_tokens=600)
        #response = response[0]['generated_text']


        prompt_pii = match_pii(prompt, full_pii)
 
        entry = {'ID':uid,
                 'Category': category,
                 'Scenario': scenario,
                 'Prompt': prompt,
                 'PromptPII': prompt_pii,
                 'Content':content}

        entries.append(entry)
        del prompt


dataset = Dataset.from_list(entries)
dataset.save_to_disk(f"PANOPTICON_{chunk_index}")
dataset.to_json(f'PANOPTICON_{chunk_index}.JSON', lines=True)