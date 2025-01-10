import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Zero-Shot Classification from transformers
from transformers import pipeline

# ----------------------------------------------------------------------
# 1) READ CSV FILE & DEFINE LABELS FOR COMMIT TYPE
# ----------------------------------------------------------------------
csv_file_path = 'git_history.csv'  # Updated to use correct file
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

# ----------------------------------------------------------------------
# 2) INSTANTIATE CLASSIFIER A FOR COMMIT TYPE
# ----------------------------------------------------------------------
classifier_a = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ----------------------------------------------------------------------
# 3) FUNCTION TO CATEGORIZE COMMITS
# ----------------------------------------------------------------------
def categorize_commit(message, code_changes):
    """
    Use Zero-Shot Classification (Classifier A) on the commit message + code changes
    to determine the commit type among possible_commit_types.
    """
    text_input = f"{message}\n{code_changes}"

    result = classifier_a(
        text_input,
        candidate_labels=possible_commit_types,
        multi_label=False  # pick a single best label
    )
    best_label = result["labels"][0]
    return best_label

# Apply commit-type classification
df['Category'] = df.apply(
    lambda x: categorize_commit(x['Message'], x['Code Changes']), 
    axis=1
)

# ----------------------------------------------------------------------
# 4) GENERATE DONUT CHART OF COMMIT CATEGORIES
# ----------------------------------------------------------------------
def generate_commit_type_donut(df):
    """
    Generate a donut chart for commit 'Category' distribution.
    Use lines/branches to annotate each slice without collision.
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
        angle = mid_angle(p.theta1, p.theta2)
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        category_label = category_counts.index[i]
        cat_count = category_counts[category_label]
        pct = 100. * cat_count / total_commits

        horizontalalignment = "right" if x < 0 else "left"

        connectionstyle = f"angle,angleA=0,angleB={angle}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        ax.annotate(
            f"{category_label}\n{pct:.1f}%",
            xy=(x, y),            # point at wedge boundary
            xytext=(1.3*np.sign(x), 1.3*y),  # text offset
            horizontalalignment=horizontalalignment,
            **kw
        )

    ax.set_title("Distribution of Git Commits by Category", pad=20)
    plt.axis('equal')

    # Save or show
    plt.savefig('commit_categories_donut.png', bbox_inches='tight', dpi=300)
    plt.close()

generate_commit_type_donut(df)

print("Donut chart generation complete!")
print("• Donut chart saved as 'commit_categories_donut.png'.")
