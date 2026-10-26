# Git Commit Guide (Industry Standards)

## 1. La règle d'or : Les Commits Atomiques
- **1 Commit = 1 seule unité logique** (une tâche, une fonction, un bugfix).
- **Le code doit toujours compiler/fonctionner** à chaque commit (ne jamais casser `main`).
- Ne mélange pas un refactoring, une nouvelle fonctionnalité et une correction de typo dans le même commit.

---

## 2. Le format : Conventional Commits
Chaque message de commit suit le format :
`<type>(<scope>): <description>`

### Les Types principaux :
| Type | Quand l'utiliser ? | Exemple |
| :--- | :--- | :--- |
| `feat` | Nouvelle fonctionnalité pour l'utilisateur / le programme | `feat(sandbox): add safe open wrapper` |
| `fix` | Correction d'un bug ou d'un comportement inattendu | `fix(security): allow files inside directories` |
| `refactor` | Modification du code sans changer le comportement | `refactor(llm): simplify prompt builder` |
| `test` | Ajout ou modification de tests unitaires | `test(sandbox): add test for path traversal` |
| `docs` | Documentation uniquement (README, docstrings) | `docs(readme): add installation guide` |
| `chore` | Tâches de maintenance (dépendances, .gitignore, CI) | `chore(deps): update uv lockfile` |

### Le Scope :
Indique la partie du projet concernée entre parenthèses :
`(sandbox)`, `(llm)`, `(mcp)`, `(agent)`, `(swebench)`...

### La Description :
1. **Langue :** Toujours en anglais.
2. **Temps :** Impératif présent (écris `add`, `fix`, `allow` — pas ~~added~~ ni ~~fixing~~).
3. **Casse :** Commence par une minuscule.
4. **Ponctuation :** Aucun point final.
5. **Longueur :** Moins de 72 caractères (court et percutant).

---

## 3. Bonnes pratiques de flow
- **Commit souvent en local :** Dès qu'une fonction passe ses tests, commit.
- **Vérifie avec `git diff` :** Avant de commiter, relis toujours tes changements avec `git diff` ou `git diff --staged`.
- **Nomme clairement :** Si tu n'arrives pas à résumer ton commit en une phrase simple, c'est probablement qu'il contient trop de choses différentes (découpe-le en plusieurs commits).