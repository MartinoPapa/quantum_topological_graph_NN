# Plan: Set LaTeX Formatter

The LaTeX Workshop extension requires a formatting tool to be configured in your workspace settings. A highly recommended formatter for LaTeX is `latexindent`.

## Steps to Take

1. **Create the Settings File:** I will create a `.vscode/settings.json` file in the root of your workspace if it doesn't already exist.
2. **Add Formatter Configuration:** I will add the following key-value pair to configure `latexindent` as the formatter:
   ```json
   {
       "latex-workshop.formatting.latex": "latexindent"
   }
   ```

> **Note:** For this formatting to work, you will need to have `latexindent` installed on your system (it usually comes with TeX Live or MiKTeX, though it may require additional Perl modules depending on your OS).

Please approve this plan so I can apply the changes!
