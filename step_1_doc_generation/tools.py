import os
import random
import time
from openai import OpenAI
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import requests
import json
import yaml
import ast
from bs4 import BeautifulSoup, Tag, NavigableString
import tiktoken

gpt_models = [
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4o",
    "gpt-4-turbo"
]
claude_models = [
    "claude-3-haiku-20240307", "claude-3-opus-20240229",
    "claude-3-sonnet-20240229"
]


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def query_model(model, prompt):
    try:
        if model in gpt_models:
            answer = query_gpt(prompt, model_name=model)
        elif model in claude_models:
            answer = query_claude(prompt, model_name=model)
        elif model == "autodroidv2":
            answer = query_autodroidv2(prompt)
        else:
            answer = query_llm(prompt, model_name=model)
        return answer
    except Exception as e:
        raise e


def query_autodroidv2(prompt: str):
    model_path = "autodroidv2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def query_gpt(prompt, model="gpt-3.5-turbo"):
    '''
    @param model:
      claude-3-opus-20240229
      gpt-3.5-turbo
      gpt-4
    '''
    max_retry = 8
    if model.startswith("gpt"):
        cli = OpenAI(base_url=os.getenv("OPENAI_API_URL"),
                     api_key=os.getenv("OPENAI_API_KEY"))
        retry = 0
        err = None
        while retry < max_retry:
            try:
                completion = cli.chat.completions.create(messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                    model=model,
                    timeout=60)
                break
            except Exception as e:
                print(f'retrying {retry} times...')
                retry += 1
                err = e
                continue

        if retry == max_retry:
            raise err

        res = completion.choices[0].message.content

    elif model.startswith("claude"):
        url = 'https://api.openai-proxy.org/anthropic/v1/messages'

        headers = {
            'x-api-key': os.getenv("OPENAI_API_KEY"),
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }

        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }

        retry = 0
        err = None
        while retry < max_retry:
            try:
                response = requests.post(url,
                                         headers=headers,
                                         data=json.dumps(data),
                                         timeout=30,
                                         )
                break
            except Exception as e:
                retry += 1
                err = e
                continue

        if retry == max_retry:
            raise err

        res = response.json()['content'][0]['text']

    return res


def load_json_file(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def dump_json_file(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f)


def debug_query_gptv2(prompt: str, model_name: str, temperature: float = 0.2, timeout: int = 120):
    client = OpenAI(
        base_url=os.getenv("OPENAI_API_URL"),
        api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        timeout=timeout,
        temperature=temperature
    )
    res = completion.choices[0].message.content
    return res


def escape_xml_chars(input_str):
    """
    Escapes special characters in a string for XML compatibility.

    Args:
        input_str (str): The input string to be escaped.

    Returns:
        str: The escaped string suitable for XML use.
    """
    if not input_str:
        return input_str
    return (input_str.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def load_txt_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def write_txt_file(txt_path, data):
    with open(txt_path, 'w') as f:
        f.write(data)


def write_txt_file(txt_path, data):
    with open(txt_path, 'w', encoding="utf-8") as f:
        f.write(data)


def load_yaml_file(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def dump_yaml_file(yaml_path, data):
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


def load_jsonl_file(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def dump_jsonl_file(dict_list, filename):
    with open(filename, 'w') as file:
        for dict_item in dict_list:
            json_str = json.dumps(dict_item)
            file.write(json_str + '\n')


def extract_common_structurev0(html1, html2):
    '''
    Extract the common structure from two HTML strings, return the string representation of the common structure and the Tag object of the common structure.
    '''

    def compare_and_extract_common(node1, node2):
        if not (node1 and node2):
            return None
        if node1.name != node2.name:
            return None
        common_node = Tag(name=node1.name)
        for attr in node1.attrs:
            if attr in node2.attrs and node1.attrs[attr] == node2.attrs[attr]:
                common_node[attr] = node1.attrs[attr]
        common_children = []
        for child1, child2 in zip(node1.find_all(recursive=False), node2.find_all(recursive=False)):
            common_child = compare_and_extract_common(child1, child2)
            if common_child:
                common_children.append(common_child)
        common_node.extend(common_children)
        return common_node

    soup1 = BeautifulSoup(html1, 'html.parser')
    soup2 = BeautifulSoup(html2, 'html.parser')

    common_structure = compare_and_extract_common(soup1.contents[0], soup2.contents[0])
    return common_structure.prettify(), common_structure


def remove_ids(html: str):
    '''
    use bs4 to remove all id attribute from html description
    '''

    def _remove_ids_from_soup(soup):
        for tag in soup.find_all(True):
            if 'id' in tag.attrs:
                del tag.attrs['id']
        return soup

    soup = BeautifulSoup(html, 'html.parser')
    cleaned_soup = _remove_ids_from_soup(soup)
    return cleaned_soup.prettify()


def clean_repeated_siblings(html: str):
    '''
    use bs4 to remove all repeated siblings from the html
    '''

    def _remove_repeated_siblings(tag):
        if not isinstance(tag, Tag):
            return
        unique_children = []
        seen_tags = set()
        for child in tag.find_all(recursive=False):
            child_signature = (child.name, tuple(sorted(child.attrs.items())))
            if child_signature not in seen_tags:
                unique_children.append(child)
                seen_tags.add(child_signature)
        tag.clear()
        for child in unique_children:
            tag.append(child)
            _remove_repeated_siblings(child)

    soup = BeautifulSoup(html, 'html.parser')
    _remove_repeated_siblings(soup)
    return soup.prettify()


def clean_attributes(html: str):
    '''
    use bs4 to remove all other attributes except for the tag name and resource_id from the html
    '''

    def _clean_attributes_from_soup(soup):
        for tag in soup.find_all(True):
            # Remove all attributes except for 'resource_id'
            attributes_to_keep = {'resource_id'}
            attributes = tag.attrs.copy()
            for attr in attributes:
                if attr not in attributes_to_keep:
                    del tag.attrs[attr]

            # Remove all text nodes within tags
            for content in tag.contents:
                if isinstance(content, NavigableString):
                    content.extract()
        return soup

    soup = BeautifulSoup(html, 'html.parser')
    cleaned_soup = _clean_attributes_from_soup(soup)
    return cleaned_soup.prettify()


def extract_common_structure(html1, html2, clean_redundant_attributes=False, clean_siblings=True):
    if clean_redundant_attributes:
        html1 = clean_attributes(html1)
        html2 = clean_attributes(html2)
    if clean_siblings:
        html1 = clean_repeated_siblings(html1)
        html2 = clean_repeated_siblings(html2)
    soup1 = BeautifulSoup(html1, 'html.parser')
    soup2 = BeautifulSoup(html2, 'html.parser')

    def compare_elements(elem1, elem2):
        if elem1.name != elem2.name:
            return False
        if elem1.attrs.get('resource_id') != elem2.attrs.get('resource_id'):
            return False
        return True

    def find_common_elements(parent1, parent2):
        common_elements = []
        for child1 in parent1.find_all(recursive=False):
            for child2 in parent2.find_all(recursive=False):
                if compare_elements(child1, child2):
                    common_elements.append((child1, child2))
        return common_elements

    def build_common_layout(common_elements):
        if not common_elements:
            return None
        common_layout = BeautifulSoup('', 'html.parser')
        for elem1, elem2 in common_elements:
            new_elem = BeautifulSoup(str(elem1), 'html.parser').find()
            new_elem.clear()
            sub_common_elements = find_common_elements(elem1, elem2)
            sub_common_layout = build_common_layout(sub_common_elements)
            if sub_common_layout:
                new_elem.append(sub_common_layout)
            common_layout.append(new_elem)
        return common_layout

    common_elements = find_common_elements(soup1, soup2)
    common_layout = build_common_layout(common_elements)
    if not common_layout:
        return None, None
    return common_layout.prettify(), common_layout


def count_ele_num(tag: Tag):
    '''
    count the number of elements in the tag
    '''
    count = 1
    for child in tag.children:
        if isinstance(child, Tag):
            count += count_ele_num(child)
    return count


def _convert_str_to_json(input_str):
    try:
        converted_answer = ast.literal_eval(input_str)
        return converted_answer
    except:
        converted_answer = json.loads(input_str)
        return converted_answer


def convert_gpt_answer_to_json(answer, model_name, default_value={'default': 'format wrong'},
                               query_func=debug_query_gptv2):
    import ast
    convert_prompt = f'''
Convert the following data into JSON dict format. Return only the dict. Ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).

data:
{answer}

**Please do not output any content other than the JSON dict format.**
'''
    try:
        answer = answer.replace('```json', '').replace('```dict', '').replace('```list', '').replace('```python', '')
        answer = answer.replace('```', '').strip()

        # converted_answer = ast.literal_eval(answer)
        converted_answer = _convert_str_to_json(answer)

    except:
        print('*' * 10, 'converting', '*' * 10, '\n', answer, '\n', '*' * 50)
        converted_answer = query_func(convert_prompt, model_name)
        print('*' * 10, 'converted v1', '*' * 10, '\n', converted_answer, '\n', '*' * 10)
        if isinstance(converted_answer, str):
            try:
                converted_answer = converted_answer.replace('```json', '').replace('```dict', '').replace('```list',
                                                                                                          '').replace(
                    '```python', '')
                converted_answer = converted_answer.replace('```', '').strip()
                # converted_answer = ast.literal_eval(converted_answer)
                converted_answer = _convert_str_to_json(converted_answer)
            except:
                new_convert = f'''
Convert the following data into JSON dict format. Return only the JSON dict. Ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).
data:
{answer}

The former answer you returned:
{converted_answer}
is wrong and can not be parsed in python. Please check it and convert it properly!

**Please do not output any content other than the JSON dict format!!!**
'''
                converted_answer = query_func(new_convert, model_name)
                print('*' * 10, 'converted v2', '*' * 10, '\n', converted_answer, '\n', '*' * 10)
                if isinstance(converted_answer, str):
                    try:
                        converted_answer = converted_answer.replace('```json', '').replace('```dict', '').replace(
                            '```list', '').replace('```python', '')
                        converted_answer = converted_answer.replace('```', '').strip()
                        # converted_answer = ast.literal_eval(converted_answer)
                        converted_answer = _convert_str_to_json(converted_answer)
                    except:
                        return default_value
    return converted_answer


def get_combined_code(pre_code_path, code):
    preparation_code = load_txt_file(pre_code_path)
    combined_code = preparation_code + '\n' + code
    return combined_code


def get_code_without_prefix(pre_code_path, code):
    preparation_code = load_txt_file(pre_code_path)
    stripped_code = code.replace(f'{preparation_code}\n', '')
    return stripped_code


def get_leading_tabs(string):
    '''
    extract the tabs at the beginning of a string
    '''
    space_num = len(string) - len(string.lstrip(' '))
    tabs_num = len(string) - len(string.lstrip('\t'))
    return space_num * ' ' + tabs_num * '\t'


def get_all_error_file_names(dir_path):
    json_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".json") and 'error_' in file:
                json_files.append(os.path.join(root, file))
    return json_files


def write_dict_to_txt(file_path, data):
    dict_str = ''
    for k, v in data.items():
        dict_str += f'{k}: \n' + '=' * 100 + f'\n{v}' + '\n' + '-' * 100 + '\n'
    write_txt_file(file_path, dict_str)


def load_txt_to_dict(file_path):
    def remove_n(string):
        while string[0] == '\n' or string[0] == ' ':
            string = string[1:]
        while string[-1] == '\n' or string[-1] == ' ':
            string = string[:-1]
        return string

    result_dict = {}
    dict_str = load_txt_file(file_path)
    k_v_pairs = dict_str.split('\n' + '-' * 100 + '\n')
    for k_v in k_v_pairs:
        k_v_pair = k_v.split(': \n' + '=' * 100)

        if len(k_v_pair) < 2:
            continue
        result_dict[remove_n(k_v_pair[0])] = remove_n(k_v_pair[1])
    return result_dict


def convert_json_to_str(json_data):
    return json.dumps(json_data, ensure_ascii=False, indent=4)


def write_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data, ensure_ascii=False) + '\n')


def load_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


def safe_get_value(dictionary, key, default=None):
    if isinstance(key, list):
        for k in key:
            if k in dictionary:
                return dictionary[k]
    elif isinstance(key, str):
        if key in dictionary:
            return dictionary[key]
    return default


def query_claude(prompt: str,
                 model_name="claude-3-haiku-20240307",
                 retry_times=6):
    client = anthropic.Anthropic(base_url=os.environ.get('ANTHROPIC_API_URL'),
                                 api_key=os.environ.get('ANTHROPIC_API_KEY'))
    retry = 0
    while retry < retry_times:
        try:
            message = client.messages.create(model=model_name,
                                             max_tokens=4096,
                                             messages=[{
                                                 "role": "user",
                                                 "content": prompt
                                             }])
            res = message.content[0].text
            break
        except Exception as e:
            retry += 1
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
            print(f'retrying {retry} times...')
            err = e
        else:
            raise err

    print(f'Claude answer: {res}')
    return res


def query_llm(prompt: str, model_name="", retry_times=6):
    openai.base_url = os.environ.get('OPENAI_API_URL')
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # create a completion
    retry = 0
    err = None
    while retry < retry_times:
        try:
            completion = openai.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=4096)

            # create a chat completion
            completion = openai.chat.completions.create(model=model_name,
                                                        messages=[{
                                                            "role": "user",
                                                            "content": prompt
                                                        }],
                                                        timeout=120)
            res = completion.choices[0].message.content
            break
        except Exception as e:
            retry += 1
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
            print(f'retrying {retry} times...')
            err = e
        else:
            raise err

    print(f'{model_name} answer: {res}')
    return res


if __name__ == '__main__':
    print(query_gpt('hello world', 'gpt-3.5-turbo'))