# scientific-writing add-ons (backup)

Durable backup of customizations merged into the `claude-scientific-writer`
plugin's `scientific-writing` skill. The plugin lives in a version-hashed cache
directory, so its edits are lost when the plugin is updated/reinstalled. Keep
this copy in the repo and re-apply after any plugin update.

## Contents

- `submission_and_rebuttal.md` — Submission, editorial assessment, rebuttal, and
  appeals guidance (Nature-family workflow): article-type selection
  (Article/Letter/Brief Communication/Resource), title/abstract as reviewer
  recruitment tools, editorial triage criteria, point-by-point rebuttal
  construction, and when/how to appeal a rejection.

## How to re-apply after a plugin update

1. Locate the current plugin skill dir (the hash changes on update):
   ```bash
   ls -d ~/.claude/plugins/cache/claude-scientific-writer/claude-scientific-writer/*/skills/scientific-writing
   ```
2. Copy the reference file back in:
   ```bash
   SW=$(ls -d ~/.claude/plugins/cache/claude-scientific-writer/claude-scientific-writer/*/skills/scientific-writing)
   cp submission_and_rebuttal.md "$SW/references/"
   ```
3. Re-add the two pointers in `$SW/SKILL.md` (if not already present):
   - A section titled **"Submission, Editorial Assessment, Rebuttal, and Appeals"**
     after the "Stage 4: Final Preparation" block.
   - A bullet in the bottom **References** list:
     `` - `references/submission_and_rebuttal.md`: Article-type selection, editorial assessment criteria, point-by-point rebuttal construction, and when/how to appeal a rejection ``

Source of this content: added 2026-06-08.
