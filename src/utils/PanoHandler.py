import datasets
from dataclasses import dataclass, field
from typing import List
import random

'''
Dealing with the following datasets:

 srirxml/PANORAMA
Features {
            id: String
            content-type: String
            text: string}

content-types [
                Article,
                Social Media
                Forum Post,
                Online Review,
                Blog/News Article Comment,
                Online Ad
                ]

Splits = ['train']
num rows = 384_794
Config names = ['default']

Example entry
    id: 4f7dc93c-14bf-4f47-bd89-84a36a7e1f62

    content-type: Social Media

    text: Raymond : Just wrapped up a productive morning 
    at WOBN Trust Inc. at 68 and still full of energy! 
    Engineering never stops 🚀 #EngineeringLife 
    #NeverTooOld


    
 Dataset: srirxml/PANORAMA-Plus
    Num of Features: 36
    Splits: ['train']
    rows: 9674
    configs: ['default']

    'en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 
    'en_IE', 'en_IN', 'en_PH'
'''

@dataclass
class Panorama:
    name: str = 'srirxml/PANORAMA'
    features: List[str] = field(default_factory=lambda: ['id', 'content-type', 'text'])
    content_types: list[str] = field(default_factory=lambda: ['Article', 'Social Media', 'Forum Post', 'Online Review', 'Blog/News Article Comment', 'Online Ad'])
    split: str = 'train' 
    num_rows: int = 384_794 
    conf_name: str = 'default' 
    dataset: object = field(init=False)
    articles: list[dict] = field(init=False)
    socials: list[dict] = field(init=False)
    forums: list[dict] = field(init=False)
    reviews: list[dict] = field(init=False)
    comments: list[dict] = field(init=False)
    ads: list[dict] = field(init=False)
    

    def __post_init__(self):
        self.dataset = datasets.load_dataset(self.name, self.conf_name, split=self.split)
        self.articles = self.dataset.filter(lambda x:x['content-type'] == self.content_types[0])
        self.socials = self.dataset.filter(lambda x:x['content-type'] == self.content_types[1])
        self.forums = self.dataset.filter(lambda x:x['content-type'] == self.content_types[2])
        self.reviews = self.dataset.filter(lambda x:x['content-type'] == self.content_types[3])
        self.comments = self.dataset.filter(lambda x:x['content-type'] == self.content_types[4])
        self.ads = self.dataset.filter(lambda x:x['content-type'] == self.content_types[5])
    
@dataclass
class Entry:
    idx: int = None
    id: str = field(init=False)
    content_type: str = field(init=False)
    text: str = field(init=False)
    dataset: datasets = field(repr=None, default=Panorama().dataset)

    def __post_init__(self):
        self.id = self.dataset[self.idx]['id']
        self.content_type = self.dataset[self.idx]['content-type']
        self.text = self.dataset[self.idx]['text']

@dataclass
class PanoPlus:
    name: str = 'srirxml/PANORAMA-Plus'

    features: list[str] = field(default_factory=lambda: ['Unique ID', 'Locale', 'First Name', 'Last Name', 
                                                        "Father's Name", "Mother's Name", 'Gender', 'Age', 
                                                        'Nationality', 'Marital Status', 'Spouse Name', 
                                                        'Children Count', 'National ID', 'Passport Number', 
                                                        "Driver's License", 'Phone Number', 'Work Phone', 
                                                        'Address', 'Email Address', 'Work Email', 'Birth Date', 
                                                        'Birth City', 'Education Info', 'Finance Status', 
                                                        'Net Worth', 'Employer', 'Job Title', 'Annual Salary', 
                                                        'Credit Score', 'Social Media Handles', 'Blood Type', 
                                                        'Allergies', 'Disability', 'Emergency Contact Name', 
                                                        'Emergency Contact Phone', 'complete_info'])
    
    
    split: str = 'train'
    num_rows: int = 9_674
    conf_name: str = 'default'
    dataset: object = field(init=False)

    def __post_init__(self):
        self.dataset = datasets.load_dataset(self.name, self.conf_name, split = self.split)

    def get_ids(self):
        return self.dataset['Unique ID']
    
    def rand_features(self, features):
        num_features = random.randint(2,5)
        return random.sample(features, num_features)


