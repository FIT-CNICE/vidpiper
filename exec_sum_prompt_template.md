You are tasked with creating an executive summary based solely on a collection of 20–30 notes prepared as Marp slide decks. Each note includes YAML front matter with the following fields:

- marp-theme
- title
- subtitle
- taxonomy (e.g. "AI > Computer Vision > Pose Estimation")

Additionally, the last slide of each note contains a set of detailed, takeaways-oriented insights.

**Important Instructions:**

1. **Strictly Use Provided Notes:**  
   Base your entire analysis and final executive summary only on the provided notes. Do not inject any external information or rely on your pre-trained knowledge. If information is not present in the provided materials or the background information, do not assume it.

2. **Weighting by Taxonomy Similarity:**  
   The summary should reflect that not all notes carry equal relevance. Please weigh the contributions of each note based on how similar its "taxonomy" field is to the given main topic: **{INSERT TOPIC HERE}**. Notes with higher taxonomy similarity should be given more emphasis in your analysis and conclusions.

3. **Philosophy and Strategy Explanation:**  
   Before generating the final executive summary, provide an initial section that explains your philosophy and strategy for crafting an engaging, intuitive summary. In this section, detail:
   - How you plan to leverage the provided notes.
   - Your approach to weighting notes by taxonomy similarity.
   - How you will ensure the summary resonates with the current market and technology landscape.
   
4. **Include Background Information:**  
   Consider the following background information for additional context:
   - **Company Background:** {INSERT COMPANY BACKGROUND INFORMATION HERE}
   - **Relevant Events:** {INSERT RELEVANT EVENTS OR CONTEXTUAL DETAILS HERE}
   
   Integrate this background context where appropriate, ensuring that the summary remains strictly based on both the notes and the provided background.

5. **Output in Quarto Format:**  
   Your final output must be formatted in Quarto. This means:
   - Use Markdown for the overall structure.
   - Embed Python code snippets for generating static plots (e.g., using matplotlib) where needed.
   - Embed Observable JS code snippets for interactive visualizations.
   - Ensure that all code snippets are correctly formatted and placed within appropriate code fences so that the final document can be rendered seamlessly on a GitHub page.

6. **Controlling Hallucinations and Bias:**  
   Be mindful to:
   - Rely strictly on the supplied notes and background details.
   - Verify that every inference or conclusion is directly supported by the provided information.
   - Avoid introducing any pre-trained bias or extraneous data that isn’t evident in the inputs.
   - If in doubt or if the provided data is ambiguous, note the uncertainty rather than assuming additional facts.

**Output Structure:**

Please structure your output as follows:

---
### I. Philosophy and Strategy
- Explain your approach to creating the summary.
- Describe how you will weigh the notes based on taxonomy similarity and why this method ensures relevance to the current market/tech landscape.

### II. Analysis of Provided Notes
- Summarize the key themes and takeaways from the notes.
- Highlight how the weighting (based on taxonomy similarity) influenced your analysis.

### III. Background Context
- Briefly present the company background and relevant events as provided.
- Indicate how this context supports the overall narrative of the executive summary.

### IV. Executive Summary (Quarto Format)
- Provide the comprehensive executive summary here.
- Use Quarto syntax, including:
  - **Static Visualizations:** Embed Python code snippets for static plots.
  - **Interactive Visualizations:** Embed Observable JS code snippets.
- Ensure that all elements (text, code, plots) are integrated seamlessly.

### V. Conclusion
- Summarize the main insights and takeaways.
- Reiterate the rationale behind the analysis and the weighting approach.

---

Now, please generate the final executive summary strictly following the guidelines above, using only the provided notes and background information.
