# How to Submit Team Deliverables (GitHub)

This course uses GitHub for **all submissions** after Midterm 1.

## 0) One-time setup (only the first time)
1. Install Git on your laptop.
2. Create a GitHub account (if you do not have one).
3. Install Visual Studio Code.
4. Make sure you can run Python in your terminal.

## 1) Clone the course repository (only once)
Open a terminal and run:

```bash
git clone <REPO_URL_HERE>
cd data-mining-2026
```

> Replace `<REPO_URL_HERE>` with the repository URL provided by your instructor.

## 2) Before every class: update your local copy
Always start by pulling the latest changes from `main`:

```bash
git checkout main
git pull origin main
```

## 3) Work on your team branch
Switch to your team branch (example: Team 01):

```bash
git checkout team-01
git pull origin team-01
```

If your team branch does not exist locally yet:

```bash
git checkout -b team-01
git push -u origin team-01
```

## 4) Put your deliverables in the correct folder
Every session has a standard submission path:

```text
teams/team-XX/session_YY/
```

Example for Session 13:

```text
teams/team-01/session_13/decision_tree_structure.py
```

**Do not** submit files outside your team folder unless the instructor explicitly asks for it.

## 5) Run your code locally
Before submitting, run your script from the repository root:

```bash
python teams/team-01/session_13/decision_tree_structure.py
```

If you used a virtual environment, activate it first.

## 6) Commit your changes
Stage and commit with a clear message:

```bash
git add .
git commit -m "Session 13: decision tree structure deliverable"
```

## 7) Push your branch to GitHub
```bash
git push origin team-01
```

## 8) Open a Pull Request (PR) to submit
On GitHub:
1. Go to the repository page.
2. Click **Pull requests**.
3. Click **New pull request**.
4. Base branch: `main`
5. Compare branch: `team-01`
6. Title: `Team 01 — Session 13 Deliverable`
7. Description: include:
   - What you implemented
   - The root split and your interpretation
   - Any issues you encountered

Then click **Create pull request**.

## 9) Common mistakes to avoid
- Pushing directly to `main`
- Submitting in the wrong folder
- Forgetting to `git pull origin main` before starting
- Submitting code that does not run

If you are stuck, take a screenshot of your terminal and ask your instructor.
