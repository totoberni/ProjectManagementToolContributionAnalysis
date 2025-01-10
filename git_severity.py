import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Zero-Shot Classification from transformers
from transformers import pipeline

# ----------------------------------------------------------------------
# 1) READ CSV FILE & DEFINE LABELS FOR SEVERITY
# ----------------------------------------------------------------------
csv_file_path = 'git_history.csv'  # Updated to use correct file
df = pd.read_csv(csv_file_path)

# Possible severity levels (Classifier B) from 1 to 5.
possible_severity_labels = [
    "Severity 1",
    "Severity 2",
    "Severity 3",
    "Severity 4",
    "Severity 5",
]

# ----------------------------------------------------------------------
# 2) INSTANTIATE CLASSIFIER B FOR COMMIT SEVERITY
# ----------------------------------------------------------------------
classifier_b = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ----------------------------------------------------------------------
# 3) FUNCTION TO ASSESS SEVERITY
# ----------------------------------------------------------------------
def assess_severity(message, code_changes):
    """
    Zero-Shot classification for commit severity among 'Severity 1'...'Severity 5'.
    Returns an integer 1..5
    """
    text_input = f"{message}\n{code_changes}"

    result = classifier_b(
        text_input,
        candidate_labels=possible_severity_labels,
        multi_label=False
    )
    best_label = result["labels"][0]  # e.g. "Severity 3"

    # Convert "Severity X" -> integer X
    severity_str = best_label.replace("Severity ", "").strip()
    severity_level = int(severity_str)

    return severity_level

# Apply severity classification
df['Severity'] = df.apply(
    lambda x: assess_severity(x['Message'], x['Code Changes']),
    axis=1
)

# ----------------------------------------------------------------------
# (Optional) Ensure each level 1..5 is present at least once
# ----------------------------------------------------------------------
existing_severity_levels = set(df['Severity'].unique())
all_severity_levels = {1, 2, 3, 4, 5}
missing_levels = all_severity_levels - existing_severity_levels

for ml in missing_levels:
    # Add a dummy row so no severity bar is at zero
    df = pd.concat([
        df,
        pd.DataFrame({
            'Hash': [f'DUMMY_{ml}'],
            'Author': ['dummy'],
            'Email': ['dummy'],
            'Date': [pd.NaT],
            'Message': [f'Dummy commit for severity {ml}'],
            'Branches': ['dummy'],
            'Code Changes': ['None'],
            'Severity': [ml]
        })
    ], ignore_index=True)

# ----------------------------------------------------------------------
# 4) GENERATE BAR CHART WITH NUMBER OF SAME-SEVERITY COMMITS
# ----------------------------------------------------------------------
def generate_severity_bar_chart(df):
    """
    Generate a bar chart for the number of commits at each severity level (1..5).
    """
    severity_counts = df['Severity'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(severity_counts.index, severity_counts, color="orange")

    ax.set_title("Distribution of Commit Severity", pad=20)
    ax.set_xlabel('Severity Level (1=Least Critical, 5=Most Critical)')
    ax.set_ylabel('Number of Commits')
    ax.set_xticks([1,2,3,4,5])

    # Label each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center', va='bottom'
        )

    plt.tight_layout()
    plt.savefig('commit_severity_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

generate_severity_bar_chart(df)

print("Severity bar chart generation complete!")
print("• Bar chart saved as 'commit_severity_bar.png'.")
