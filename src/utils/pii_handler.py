from random import sample, randint
import re
from ast import literal_eval

#Regex help
basic_text = re.compile(r"[A-Za-z+-]+")
basic_number = re.compile(r'[0-9]+')
alphanumeric = re.compile(r"[A-Za-z0-9]+")
email = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
birthday = re.compile(
    r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
)
money = re.compile(r'[0-9$£€₹₱SR]+(?:\.\d+)')

phone_number = re.compile(r'\+?\d[\d\-\s\(\)]{4,}\d')

# Common filler/stop tokens to avoid matching benign overlaps (esp. addresses)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in', 'into',
    'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'were',
    'with', 'your', 'my', 'our', 'their', 'his', 'her', 'its', 'these', 'those',
    'after', 'before', 'between', 'during', 'about', 'over', 'under', 'out',
    'up', 'down', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w', 'way',
    'road', 'street', 'st', 'rd', 'ave', 'avenue', 'blvd', 'boulevard', 'lane',
    'ln', 'drive', 'dr', 'court', 'ct', 'circle', 'cir', 'place', 'pl',
    'parkway', 'pkwy', 'trail', 'trl', 'highway', 'hwy', 'route', 'unit', 'apt',
    'suite', 'floor', 'fl'
}

def normalize_phone(p: str) -> str:
    p = p.strip()

    p = re.sub(r'\([^)]*\)\s*$', '', p).strip()


    plus = p.startswith('+')
    digits = ''.join(ch for ch in p if ch.isdigit())
    if not digits:
        return ''
    return ('+' if plus else '') + digits

address_pattern = re.compile(r"[A-Za-z0-9']+")


def _tokenize_words(text: str) -> list[str]:
    return re.findall(address_pattern, text)


def _phrase_in_prompt(value: str, prompt: str) -> bool:
    """Case-insensitive whole-word/phrase check with boundaries."""
    if not value:
        return False
    try:
        return bool(re.search(rf'\b{re.escape(value)}\b', prompt, re.IGNORECASE))
    except re.error:
        return False


def _partial_match(value: str, prompt_tokens: set[str], *, require_two: bool = False, allow_digits: bool = True) -> str:
    """Return meaningful overlapping tokens, filtering stopwords and weak singles."""
    tokens = _tokenize_words(value)
    matched = []
    strong = 0

    for tok in tokens:
        low_tok = tok.lower()
        if low_tok in STOPWORDS:
            continue
        if not allow_digits and tok.isdigit():
            continue
        if low_tok in prompt_tokens:
            matched.append(tok)
            # digits or longer tokens are stronger signals
            if tok.isdigit() or len(low_tok) >= 5:
                strong += 1

    if require_two:
        # Accept if we have 2+ tokens or at least one strong token
        if len(matched) >= 2 or strong:
            return ' '.join(matched)
        return ''

    return ' '.join(matched) if matched else ''

child_map = {'1':'one','2':'two', '3':'three', '4':'four'}
#6 pools (+ benign handled elsewhere). All relevant PII attributes from PANORAMA (excluding Unique ID, Locale, and Complete Info as potential attributes)
professional_pii = ['First Name', 'Last Name', 'Work Email', 'Employer', 'Job Title', 'Annual Salary', 'Education Info', 'Work Phone', 'Address',  'Age'] #10
finance_pii = ['Annual Salary', 'Net Worth', 'Finance Status', 'Credit Score', 'Email Address', 'Spouse Name', 'Marital Status', 'Children Count', 'Work Phone', 'Employer'] #10
health_pii = ['Age', 'Gender', 'Blood Type', 'Allergies', 'Disability' ,'Emergency Contact Name', 'Emergency Contact Phone', 'Address', 'Birth Date', 'Children Count'] #10
relationship_pii = ['First Name', 'Last Name', 'Marital Status', 'Spouse Name', "Father's Name", "Mother's Name", "Passport Number", 'Children Count', 'Phone Number', 'Gender'] #10
government_pii = ['First Name', 'Last Name', 'National ID', 'Passport Number', 'Driver\'s License', 'Birth Date', 'Address', 'Phone Number', 'Nationality', 'Employer', 'Age'] #11
social_pii = ['Social Media Handles', 'First Name', 'Last Name', 'Spouse Name', 'Nationality', 'Email Address', 'Phone Number', "Father's Name", "Mother's Name", 'Age', 'Marital Status','Birth City', 'Education Info'] #13

pii_pools = [professional_pii, finance_pii, health_pii, relationship_pii, government_pii, social_pii]





def sample_pii(pool: int, full_pii: dict) -> dict[str: str]:
    potential_pii = sample(pii_pools[pool], randint(2,5)) #2-5 random PII attributes
    used_pii = {}
    for pii in potential_pii:
            used_pii[pii] = full_pii[pii]
    return used_pii


def match_pii(og_prompt: str, pii_values: dict) -> dict:
    prompt_tokens = set(tok.lower() for tok in _tokenize_words(og_prompt))
    pii = {}

    for k, v in pii_values.items():
        # Simple text fields (whole phrase if possible; otherwise strong token overlap)
        if k in ('First Name', 'Last Name', 'Gender', 'Nationality', 'Marital Status', 'Finance Status', 'Birth City', 'Disability', 'Employer', 'Education Info', 'Allergies', 'Job Title', 'Blood Type'):
            if _phrase_in_prompt(v, og_prompt) or _partial_match(v, prompt_tokens):
                pii[k] = v

        # Multi-part names: require at least two parts or one strong token to avoid lone benign words
        elif k in ("Father's Name", "Mother's Name", 'Spouse Name', 'Emergency Contact Name'):
            match_val = _partial_match(v, prompt_tokens, require_two=True, allow_digits=False)
            if match_val:
                pii[k] = match_val

        elif k in ('Age', 'Credit Score'):
            prompt_nums = re.findall(basic_number, og_prompt)
            if v in prompt_nums:
                pii[k] = v

        elif k == 'Children Count':
            if any(word in og_prompt for word in ('child', 'Child', 'children', 'Children', 'kid', 'Kid', 'Kids', 'kids')):
                pii[k] = v

        elif k in ('National ID', 'Passport Number', "Driver's License"):
            prompt = re.findall(alphanumeric, og_prompt)
            if v in prompt:
                pii[k] = v

        elif k in ('Phone Number', 'Work Phone', 'Emergency Contact Phone'):
            matches = re.findall(phone_number, og_prompt)
            normalized_matches = [normalize_phone(m) for m in matches]
            v_norm = normalize_phone(v)
            if v_norm and v_norm in normalized_matches:
                pii[k] = v

        elif k in ('Email Address', 'Work Email'):
            matches = re.findall(email, og_prompt)
            if v in matches:
                pii[k] = v

        elif k == 'Birth Date':
            prompt = re.findall(birthday, og_prompt)
            if v in prompt:
                pii[k] = v

        elif k in ("Annual Salary", "Net Worth"):
            prompt = og_prompt.replace(',', '')
            prompt = re.findall(money, prompt)
            if v in prompt:
                pii[k] = v

        # Complex Cases
        elif k == 'Address':
            # Need at least two meaningful overlaps or one strong (number/long) token to accept
            val = _partial_match(v, prompt_tokens, require_two=True)
            if val:
                pii[k] = val

        elif k == 'Birth City':
            val = _partial_match(v, prompt_tokens)
            if val:
                pii[k] = val

        elif k == 'Social Media Handles':
            if pii_values[k] == '{}':
                continue
            handles = literal_eval(v)
            for platform, handle in handles.items():
                if _phrase_in_prompt(handle, og_prompt):
                    pii[platform] = handle

    return pii

if __name__ == '__main__':
    #Testing
    pii = sample_pii(professional_pii, {'First Name': 'Walter', 'Last Name': 'White', 'Work Email':'walter.white@tntech.edu', 'Job Title': 'cook', 'Annual Salary': '$1,000,000', 'Education Info':'Master\'s', 'Work Phone':'123 456 789', 'Address':'308 Negra Arroyo Lane', 'Birth Date': '12/12/1980'})
    print(pii)
