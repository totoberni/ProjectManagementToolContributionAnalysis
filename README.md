# Git Contribution Analysis Documentation

## Project Context

This documentation covers a set of Python scripts designed to analyze Git contributions for a group coursework project developing an Event Management Tool. The tool was built using Azure serverless web applications and CosmosDB database. These scripts provide quantitative insights into team contribution patterns and commit characteristics without requiring full database access.

## System Overview

The analysis pipeline consists of three main components that work sequentially:

1. **Git History Extraction** - Scrapes raw commit data from the repository
2. **Commit Classification** - Uses machine learning to categorize commits
3. **Visualization Generation** - Creates charts showing contribution patterns

## Script Components

### 1. get_git_history.py

**Purpose:** Extracts complete Git history from the local repository and exports it to CSV format.

**Key Functions:**
- Retrieves all commits including merges across all branches
- Captures commit metadata: hash, author, email, timestamp, message
- Includes code change statistics for each commit
- Shows branch information for tracking feature development

**Output:** `git_history.csv` containing structured commit data with columns:
- Hash, Author, Email, Date, Message, Branches, Code Changes

**Technical Notes:**
- Uses subprocess to execute Git commands directly
- Includes progress bar via tqdm for better UX on large repositories
- Handles encoding properly for international commit messages

### 2. git_doughnut.py

**Purpose:** Analyzes commit types and generates a donut chart showing distribution of different development activities.

**Classification Categories:**
- Bug Fix
- Events CRUD (Create, Read, Update, Delete operations for event management)
- Tickets CRUD (Ticket management operations)
- Locations CRUD (Venue/location management)
- Login CRUD (Authentication system changes)
- Merge Operation
- Testing
- Code Quality
- Refactoring
- Documentation
- Other

**Technical Implementation:**
- Uses Facebook's BART-large-MNLI model for zero-shot classification
- Combines commit message and code changes for better context
- No training required - leverages pre-trained language understanding
- Outputs annotated donut chart with percentage breakdowns

**Output:** `commit_categories_donut.png`

### 3. git_severity.py

**Purpose:** Assesses the criticality of commits and visualizes severity distribution.

**Severity Scale:**
- Level 1: Least critical (minor changes, documentation)
- Level 2: Low priority improvements
- Level 3: Standard feature development
- Level 4: Important features or significant fixes
- Level 5: Critical fixes or major functionality

**Technical Implementation:**
- Same zero-shot classification approach as category analysis
- Ensures all severity levels are represented in visualization
- Adds dummy entries if certain severity levels are absent

**Output:** `commit_severity_bar.png`

### 4. git_stats.py

**Purpose:** Combined analysis script (appears to be the initial version that was later split into specialized scripts).

**Note:** This script contains both classification pipelines but was refactored into separate modules for better maintainability.

## Machine Learning Approach

The scripts use zero-shot classification, which is particularly suitable for this use case because:

1. **No Training Data Required:** The team doesn't need labeled examples of each commit type
2. **Flexible Categories:** Can easily adjust classification categories without retraining
3. **Context-Aware:** The model understands natural language patterns in commit messages
4. **Consistent Classification:** Removes human bias in categorizing commits

The classification works by:
1. Combining commit message with code change statistics
2. Passing text through transformer model
3. Comparing against predefined category labels
4. Selecting most probable category based on semantic similarity

## Usage Workflow

1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Extract Git History:**
   ```bash
   python get_git_history.py
   ```
   Generates `git_history.csv`

3. **Generate Commit Type Analysis:**
   ```bash
   python git_doughnut.py
   ```
   Creates donut chart showing what types of work were done

4. **Generate Severity Analysis:**
   ```bash
   python git_severity.py
   ```
   Creates bar chart showing commit importance distribution

## Insights for Group Assessment

These visualizations help demonstrate:

- **Work Distribution:** Shows who contributed to which aspects (CRUD operations, testing, documentation)
- **Development Patterns:** Reveals whether work focused on features vs. maintenance
- **Code Quality Focus:** Percentage of commits dedicated to refactoring and quality improvements
- **Critical Contributions:** Identifies high-severity commits that were crucial to project success

## Dependencies

- **tqdm:** Progress bar for long-running operations
- **matplotlib:** Chart generation
- **numpy/pandas:** Data manipulation
- **transformers:** Hugging Face models for NLP classification
- **torch:** PyTorch backend for transformer models

## Limitations and Considerations

1. **Classification Accuracy:** Zero-shot models make educated guesses but may misclassify ambiguous commits
2. **Severity Subjectivity:** What constitutes "critical" varies by context
3. **Merge Commits:** May skew statistics if not handled properly
4. **Code Changes:** The diff statistics don't capture code quality, only quantity

## Future Improvements

Potential enhancements for more detailed analysis:
- Time-series analysis to show project progression
- Author-specific breakdowns for individual contributions
- Integration with Azure DevOps metrics
- Correlation between commit types and project milestones
