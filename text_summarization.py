from datasets import load_dataset
from transformers import pipeline
from IPython.display import display

xsum_dataset = load_dataset(
    "xsum", version="1.2.0", cache_dir="cache"
)  # Note: We specify cache_dir to use predownloaded data.
xsum_dataset  # The printed representation of this object shows the `num_rows` of each dataset split.

xsum_sample = xsum_dataset["train"].select(range(10))
display(xsum_sample.to_pandas())

summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True,
    model_kwargs={"cache_dir": "cache"},
)  # Note: We specify cache_dir to use predownloaded models.
# Apply to 1 article
summarizer(xsum_sample["document"][0])

# Apply to a batch of articles
results = summarizer(xsum_sample["document"])

# Display the generated summary side-by-side with the reference summary and original document.
# We use Pandas to join the inputs and outputs together in a nice format.
import pandas as pd

display(
    pd.DataFrame.from_dict(results)
        .rename({"summary_text": "generated_summary"}, axis=1)
        .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)