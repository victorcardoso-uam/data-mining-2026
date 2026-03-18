# Branch Policy (Teams)

## Goal
Keep `main` clean and stable, while teams submit work safely and consistently.

## Branch naming
- Instructor branches (optional): `instructor/session-13`
- Team branches (required): `team-01`, `team-02`, ..., `team-10`
- Optional feature branches (advanced teams): `team-01/session-13`

## Workflow (teams)
1) Update your local repo:
   - `git checkout main`
   - `git pull origin main`
2) Switch to your team branch:
   - `git checkout team-01`
   - `git pull origin team-01`
3) Create your deliverable inside your team folder (see below).
4) Commit with a clear message:
   - `git add .`
   - `git commit -m "Session 13: decision tree structure deliverable"`
5) Push:
   - `git push origin team-01`
6) Open a Pull Request:
   - Base: `main`
   - Compare: your `team-01`
   - Title: `Team 01 — Session 13 Deliverable`
   - Description: what you did + key results

## Deliverable location convention
Teams submit inside:
`teams/team-XX/session_13/`

Example:
`teams/team-01/session_13/decision_tree_structure.py`

## Do not
- Push directly to `main`
- Rename session folders
- Delete instructor starter files
