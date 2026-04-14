# create and activate a venv for the project
# python -m venv venv
# source venv/bin/activate

# pip install datasets pandas

# load the gpt4 split from huggingface dataset grazh/synth-medic
from datasets import load_dataset
import pandas as pd

def load_gpt4_split(split_name):
    dataset = load_dataset("grazh/synth-medic", split=split_name)
    return dataset

# create a new column in the dataset for the category, diagnosis and document id - initialize it by splitting the id column by the first underscore and taking the first part as the category and the second part as the diagnosis
def create_category_diagnosis_columns(dataset):
    def split_id(id):
        parts = id.split('_')
        return parts[0], parts[1], parts[2] if len(parts) > 2 else None
    def len_text(text):
        return len(text)
    dataset = dataset.map(lambda x: {'category_diagnosis': split_id(x['id'])[0] + '_' + split_id(x['id'])[1], 'document_id': split_id(x['id'])[2], 'length': len_text(x['text'])})
    return dataset

if __name__ == "__main__":
    # load the gpt4 split
    gpt4_split = load_gpt4_split("gpt4")

    # create a new column in the dataset for the category, diagnosis and document id
    gpt4_split = create_category_diagnosis_columns(gpt4_split)
    # order the dataset by category_diagnosis and text length ascending
    #gpt4_split = gpt4_split.sort(lambda x: (x['category_diagnosis'], x['length']))

    # save the dataset to a tsv file
    gpt4_split.to_csv("gpt4_split.tsv", index=False, sep='\t')

    df_gpt4_split = gpt4_split.to_pandas()
    df_gpt4_split.sort_values(by=['category_diagnosis', 'length'], inplace=True)
    print(df_gpt4_split.head(30))

    # save a subset containing 30 shortest text samples per category and diagnosis in a separate folder as txt files
    # save the shortest texts per category and diagnosis to another folder as txt files

    categories = gpt4_split.unique('category_diagnosis')
    
    #make a folder for the top 30 texts and another folder for the shortest texts
    import os
    os.makedirs("texts_top_30", exist_ok=True)
    os.makedirs("texts_top_1", exist_ok=True)

    for category in categories:
        subset = gpt4_split.filter(lambda x: x['category_diagnosis'] == category)
        # sort an arrow dataset by a column and select the top n rows
        subset = subset.sort('length').select(range(min(30, len(subset))))
        # save each row in a separate file named {category}_{diagnosis}_{i}.txt where i is the document_id of the row in the subset

        for i, row in enumerate(subset):
            os.makedirs(f"texts_top_30/{category}", exist_ok=True)
            with open(f"texts_top_30/{category}/{category}_{row['document_id']}.txt", "w") as f:
                f.write(row['text'])

        # save the shortest text in a separate file named {category}_{diagnosis}_{i}_shortest.txt
        if subset:
            os.makedirs(f"texts_top_1/{category}", exist_ok=True)
            with open(f"texts_top_1/{category}/{category}_{subset[0]['document_id']}_shortest.txt", "w") as f:
                f.write(subset[0]['text'])
