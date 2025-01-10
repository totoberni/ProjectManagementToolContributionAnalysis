import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# For Zero-Shot classification
from transformers import pipeline

# ----------------------------------------------------------------------
# 1) READ CSV FILE & DEFINE POSSIBLE LABELS
# ----------------------------------------------------------------------

# Replace 'your_commits.csv' with the actual CSV path
csv_file_path = 'your_commits.csv'

df = pd.read_csv(csv_file_path)

# Possible commit types (Classifier A)
possible_commit_types = [
    "Bug Fix",
    "Events CRUD",
    "Tickets CRUD",
    "Locations CRUD",
    "Login CRUD",
    "Merge Operation",
    "Testing",
    "Code Quality",
    "Refactoring",
    "Documentation",
    "Other",
]

# Possible severity levels (Classifier B) from 1 (lowest) to 5 (highest).
# We’ll do zero-shot classification but must be mindful that these are numeric labels.
# For the zero-shot approach, we can label them textually, e.g., "Severity 1", "Severity 2", ...
# Then we can map them back to numeric levels.
possible_severity_labels = [
    "Severity 1",
    "Severity 2",
    "Severity 3",
    "Severity 4",
    "Severity 5",
]

# ----------------------------------------------------------------------
# 2) INSTANTIATE TWO CLASSIFIERS (A & B)
# ----------------------------------------------------------------------

# Classifier A for commit type
classifier_a = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Classifier B for commit severity
classifier_b = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ----------------------------------------------------------------------
# 3) REFACTOR categorize_commit() TO USE CLASSIFIER A
# ----------------------------------------------------------------------
def categorize_commit(message, code_changes):
    """
    Use Zero-Shot Classification (Classifier A) on the commit message + code changes
    to determine the commit type among possible_commit_types.
    """
    # Combine both message & code changes as input to get more context
    text_input = f"{message}\n{code_changes}"

    # Zero-shot classification
    result = classifier_a(
        text_input,
        candidate_labels=possible_commit_types,
        multi_label=False  # we pick one best label
    )

    best_label = result["labels"][0]

    return best_label

# ----------------------------------------------------------------------
# 4) REFACTOR assess_severity() TO USE CLASSIFIER B
# ----------------------------------------------------------------------
def assess_severity(message, code_changes):
    """
    Use Zero-Shot Classification (Classifier B) on the commit message + code changes
    to determine severity among 'Severity 1'...'Severity 5'.
    We'll then convert it to an integer 1..5.
    """
    text_input = f"{message}\n{code_changes}"

    result = classifier_b(
        text_input,
        candidate_labels=possible_severity_labels,
        multi_label=False
    )
    best_label = result["labels"][0]  # e.g. "Severity 3"

    # Convert best_label string to integer. 
    # E.g. "Severity 3" -> 3
    severity_str = best_label.replace("Severity ", "").strip()
    severity_level = int(severity_str)

    return severity_level

# Apply the classifiers to each commit
df['Category'] = df.apply(
    lambda x: categorize_commit(x['Message'], x['Code Changes']),
    axis=1
)
df['Severity'] = df.apply(
    lambda x: assess_severity(x['Message'], x['Code Changes']),
    axis=1
)

# ----------------------------------------------------------------------
# Before generating charts, we can ensure that each severity label (1-5)
# exists in the data. If the zero-shot model never assigned e.g. severity 5,
# we can forcibly add a "dummy" row or handle it differently.
# ----------------------------------------------------------------------
# One approach: find which severity levels are missing and add dummy rows
existing_severity_levels = set(df['Severity'].unique())
all_severity_levels = set([1, 2, 3, 4, 5])

missing_levels = all_severity_levels - existing_severity_levels
for ml in missing_levels:
    # Append a dummy row to ensure we have each severity in the final distribution
    # This row won't affect the donut chart for categories, but ensures bar chart won't have zero bars.
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
            'Category': ['Other'],     # or some neutral
            'Severity': [ml]
        })
    ], ignore_index=True)

# ----------------------------------------------------------------------
# 5) GENERATE DOUGHNUT CHART FOR COMMIT CATEGORIES
# ----------------------------------------------------------------------
def generate_commit_type_donut(df):
    """
    Generate a donut chart for the 'Category' distribution.
    Uses lines/branches to annotate each slice without collision.
    """
    category_counts = df['Category'].value_counts()
    total_commits = category_counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))

    wedges, _ = ax.pie(
        category_counts,
        wedgeprops=dict(width=0.4),  # width < 1 => donut
        startangle=0
    )

    # annotation styling
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow_style = dict(arrowstyle="-", connectionstyle="angle")
    kw = dict(arrowprops=arrow_style, bbox=bbox_props, zorder=0, va="center")

    def mid_angle(t1, t2):
        return (t2 - t1) / 2. + t1

    # Label each wedge with lines
    for i, p in enumerate(wedges):
        # find the angle
        angle = mid_angle(p.theta1, p.theta2)
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        category_label = category_counts.index[i]
        cat_count = category_counts[category_label]
        pct = 100. * cat_count / total_commits

        # alignment
        horizontalalignment = "right" if x < 0 else "left"

        # update the arrow connectionstyle
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        ax.annotate(
            f"{category_label}\n{pct:.1f}%",
            xy=(x, y),  # where the arrow points
            xytext=(1.3*np.sign(x), 1.3*y),  # label offset
            horizontalalignment=horizontalalignment,
            **kw
        )

    ax.set_title("Distribution of Git Commits by Category", pad=20)
    plt.axis('equal')

    # Save and/or show
    plt.savefig('commit_categories_donut.png', bbox_inches='tight', dpi=300)
    plt.close()

generate_commit_type_donut(df)

# ----------------------------------------------------------------------
# 6) (OPTIONAL) GENERATE BAR CHART WITH THE NUMBER OF SAME-SEVERITY COMMITS
# ----------------------------------------------------------------------
def generate_severity_bar_chart(df):
    """
    Generate a bar chart for the number of commits at each severity level.
    """
    severity_counts = df['Severity'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(severity_counts.index, severity_counts, color="orange")

    ax.set_title("Distribution of Commit Severity", pad=20)
    ax.set_xlabel('Severity Level (1=Least, 5=Most Critical)')
    ax.set_ylabel('Number of Commits')
    ax.set_xticks([1,2,3,4,5])

    # Put labels above each bar
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

# Uncomment the line below if you want to generate the bar chart in the same script:
# generate_severity_bar_chart(df)

print("Analysis complete!")
print("• Donut chart saved as 'commit_categories_donut.png'.")
print("• (Optional) Severity bar chart can be saved as 'commit_severity_bar.png'.")
