# Session 15 Activity — Decision Tree Pruning on Your Project Dataset

## Objective
Apply pruning techniques to control overfitting and improve generalization on your **team project dataset**.

## You will compare three models
1. **Baseline Tree** (no pruning, or very weak constraints)
2. **Pre-pruned Tree** (use `max_depth`, `min_samples_leaf`, etc.)
3. **Cost-Complexity Pruned Tree** (use `ccp_alpha` selected via validation/CV)

## Required steps
1. Use your project dataset (CSV).
2. Decide if your task is **classification** or **regression**.
3. Choose:
   - Target column `y`
   - Feature columns `X`
4. Run the template script and fill the TODO sections.
5. Record results in comments:
   - Train score
   - Test score
   - Best pruning settings
   - 4–6 lines of interpretation

## Deliverable (What You Must Submit)
- `teams/team-XX/session_15/decision_tree_pruning.py`
- `teams/team-XX/session_15/DATASET.csv`

## Submission (GitHub)
- Commit + push to your team branch
- Create PR to `main` using:
  - `gh pr create ...`
