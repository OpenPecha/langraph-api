# Frontend UI Guide

The project includes a comprehensive web interface for testing and demonstrating the API's capabilities. This guide explains how to use its features.

### 1. Translation

The main view is for translation.

1.  **Input Texts**: Enter the source texts you wish to translate in the main text area, one per line.
2.  **Select Options**: Choose your target language, the type of text, and the AI model you wish to use.
3.  **Translate**: Click the **Translate** button. The translations will appear in the "Output" panel in real-time as they are completed.

### 2. Glossary Extraction

Once the translation is complete, the next stage of the workflow becomes available.

1.  **Extract Glossary**: A new button, **"Extract Glossary,"** will appear below the translation results.
2.  **View Results**: Clicking this button will trigger the glossary extraction service. You will see two things happen in real-time:
    *   **Inline Highlighting**: Key terms in both the source and translated texts will be highlighted with matching colors.
    *   **Consolidated Glossary**: A new table will appear at the bottom, listing all the unique key terms that were found.

### 3. Standardization Workflow

After the glossary has been extracted, you can begin the standardization process.

#### a. Analyze

1.  **Analyze Consistency**: A new button, **"Analyze Translation Consistency,"** will appear.
2.  **Review Inconsistencies**: Clicking this button calls the `/standardize/analyze` endpoint. The results are displayed in a new "Inconsistency Report" panel. This report will show you every source term that was translated in more than one way.

#### b. Decide

1.  **Choose a Standard**: For each inconsistent term, the report will show all the different translations that were found as a list of radio buttons.
2.  **Select Your Preference**: Click the radio button next to the translation you want to set as the official standard.

#### c. Apply & Confirm

1.  **Apply Standardization**: Once you have selected a standard for at least one term, the **"Apply Standardization"** button will become active.
2.  **Review the Diff**: Clicking this button will *not* immediately apply the changes. Instead, it will stream the results and show you a "diff" preview. In the results panel, you will see the old, incorrect translations struck out in red, and the new, proposed standardized translations in green.
3.  **Confirm or Cancel**: Two new buttons, **"Confirm Changes"** and **"Cancel,"** will appear.
    *   Clicking **Cancel** will discard the proposed changes and revert the view to the original translations.
    *   Clicking **Confirm Changes** will accept the new translations. The diff view will be removed, and the results will be updated to their final, standardized state.

This step-by-step, interactive workflow gives you full control over the translation and standardization process. 