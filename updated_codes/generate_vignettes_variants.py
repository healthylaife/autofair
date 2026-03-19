from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from context_article import *
from postprocessing import aggegate_vignettes
from utils import *


def generate_variant(input: dict, mode="no_hint") -> str:
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    folder_name = f"../output/{input['outcome']}_{mode}_{input['model']}_{current_time}/"

    os.makedirs(folder_name[:-1], exist_ok=True)
    os.makedirs(f"{folder_name}content", exist_ok=True)
    os.makedirs(f"{folder_name}vignettes", exist_ok=True)

    pubmed_query = generate_pubmed_query(input["outcome"])
    articles = search_article(pubmed_query, input["count"])

    for idx, a_id in enumerate(articles):
        content = get_article(a_id)
        try:
           content = get_article(a_id)
        except Exception as e:
           print(f"Skipping article {a_id} due to error: {e}")
           continue

        if content:
            with open(f"{folder_name}content/content_{idx}_{a_id}.txt", "w+", encoding="utf8") as f:
                f.write(content)

            vignettes = generate_vignettes_variant(
                input["outcome"].split(" ")[0],
                content,
                input["model"],
                mode=mode
            )

            with open(f"{folder_name}vignettes/vignettes_{idx}_{a_id}.txt", "w+", encoding="utf8") as f:
                f.write(vignettes)
        else:
            print(f"No content retrieved for article {a_id}, skipping.")

    vignette_dir = f"{folder_name}vignettes"
    if os.listdir(vignette_dir):
        aggegate_vignettes(folder_name)
    else:
        print(f"No vignette files generated for mode={mode}; skipping aggregation.")

    return folder_name



if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""   # openai api key here

    user_input = {
        "outcome": "obesity stigma",
        "count": "10,
        "model": "gpt-4o-2024-05-13",
        "sensitive_attribute": "gender"
    }

    print("Running no_hint...")
    folder1 = generate_variant(user_input, mode="no_hint")
    print("Saved to:", folder1)

    print("Running complex_context...")
    folder2 = generate_variant(user_input, mode="complex_context")
    print("Saved to:", folder2)
