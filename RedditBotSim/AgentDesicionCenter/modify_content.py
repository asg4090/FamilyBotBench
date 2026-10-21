
# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error

# Rewrite a knowledge
import sys

sys.path.append(str(_RELEASE_ROOT / "RedditBotSim"))
from llm_client import build_openai_client

def generate_news(client, model, prompt, temperature):
    try:
        completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
        llm_return_dict = completion.choices[0].message.content
        return llm_return_dict
    except Exception as e:
        print("======Exception====== /n", e)
        return None
        


def modify_content(Claim):

    input_llm_prompt = """Rewrite the news in [News] so that the generated news has a different point of view than the original news. You need to synthesize (1),(2),(3),(4),and (5) of the information to complete the response.
    (1) Modify key factors in the news, such as time, place, event, mood, opinion, etc.
    (2) You cannot directly quote the original [news], you only need to generate your revised 
    (3) The revised news [Response] should be logically consistent.
    (4) Please think step by step, how you can modify this [News].
    (5) The news data format you generate should be the same as the original news.
    [News]: {claim}
    [Response]:"""

    client, settings = build_openai_client()
    model = settings.model
    temperature = settings.temperature
    prompt = input_llm_prompt.format(claim=Claim)
    i = 1
    while i:
        llm_return_dict = generate_news(client, model, prompt, temperature)
        if llm_return_dict != None:
            break
        else:
            continue
    return llm_return_dict


