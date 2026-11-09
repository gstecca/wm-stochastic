# AI disclosure for the CAIE revision (CAIE-D-26-05414)

## Where it goes

Elsevier asks for a short statement in its own section **at the end of the manuscript, before the References**, in the main manuscript file. The heading used by Elsevier journals is:

> **Declaration of generative AI and AI-assisted technologies in the writing process**

Some Elsevier guides use a slightly different heading ("... in the manuscript preparation process"), and some restrict AI use in writing to improving readability and language. Before submitting, check the current CAIE *Guide for Authors* on elsevier.com and copy its exact heading and template.

AI tools cannot be listed as authors or cited as authors, so Claude must not appear in the author list or the acknowledgements as a contributor.

## What was actually done with AI (keep the statement consistent with this)

Used: **Claude (Anthropic), via Claude Code**, by Giuseppe Stecca, during the revision (November 2026), to:

1. draft text for the response letter and proposed revisions of the manuscript (bounds on the big-M constants and the saturation convention; the intraday vs. interday discussion; revised Section 5.1, Section 5.2 and Conclusions text);
2. recompute Table 5 and the average VSS from the stored per-instance results, which revealed the aggregation error reported by Reviewer 1;
3. prepare the public repository: extract the instance generator from the original notebook and verify that it reproduces the instances, assemble summary tables and write the README.

The authors designed the model, the experiments and all results. The AI did not generate new experimental results; it recomputed statistics from existing outputs.

## Proposed statement (main option)

> **Declaration of generative AI and AI-assisted technologies in the writing process**
>
> During the preparation of the revised version of this work, the authors used Claude (Anthropic) to help draft parts of the revised text and of the response to the reviewers, and to assist in organising the publicly released instances, code and documentation. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Shorter alternative (closer to Elsevier's template)

> During the preparation of this work the authors used Claude (Anthropic) in order to assist in drafting and revising parts of the text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

The main option is preferred because it also covers the repository and the recomputed table. If the journal's template limits the declaration to the writing process, use the short version in the manuscript and add the sentence below to the **Data availability** section.

## Optional sentence for the Data availability section

> The instance generator, summary tables and documentation in the repository were prepared with the assistance of Claude (Anthropic); all files were checked by the authors, and the generator was verified to reproduce the published instances exactly.

## Optional note in the response letter

Not required. If you want to be transparent with the reviewers (Reviewer 3 disclosed their own use of ChatGPT), add one line at the end of the letter:

> In line with the journal's policy, the use of generative AI tools during the revision is declared at the end of the revised manuscript.

## Checklist before submission

- [ ] Co-authors agree on the wording (the statement says "the authors").
- [ ] Heading and template checked against the current CAIE Guide for Authors.
- [ ] The statement is placed before the References, in the manuscript file.
- [ ] The drafted text has been reviewed and edited by the authors.
