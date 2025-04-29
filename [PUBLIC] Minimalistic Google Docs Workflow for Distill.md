

| github\_repo | https://github.com/znah/gdocs\_distill\_template |
| :---- | :---- |
| colab | https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing\_ca.ipynb |

## \<d-article\>

# A Minimalist Google Docs Workflow for Distill.pub

The typical process of a [Distill.pub](https://distill.pub/) article creation requires all contributors and editors to be familiar with version control (specifically Git) and web development, which sets an unnecessarily high barrier for entry to accept edits and suggestions from reviewers. This repository contains an article template and instructions for setting up a workflow based on Google Docs which exports the article HTML to GitHub. 

The workflow also uses a minimalistic Python web-server that automatically fetches new changes from GitHub and recompiles the article to serve the draft locally. We also provide a minimal Jekyll configuration that allows you to see changes live on GitHub Pages. 

This workflow was organically developed while writing our first Distill article \[\[mordvintsev2020growing\]\] and noticing that we really needed the collaborative functionality of Google Docs, but wanted a live preview of our article while working.

## Getting Started

* Fork and rename the [https://github.com/znah/gdocs\_distill\_template](https://github.com/znah/gdocs_distill_template) repository. Convention is to name it "post--*canonicalized-article-name*".   
* Duplicate this Google Document for your article. This also duplicates the attached Google Script allowing export to GitHub.  
* Update *github\_repo* variable in above table to point to your GitHub repository.  
* Click "HTML Export" \-\> "Run Export". You will be prompted for a GitHub API token the first time you run this. This is privately stored for your Google account and not accessible by anyone else opening the Google Doc. Give the token “repo” permissions. Get the token [here](https://github.com/settings/tokens).

![][image1]

## \<figure\>

## \<img src='export.png' style='width: 500px'\>

## \<figcaption\>A sample image demonstrating how to export html to GitHub.\</figcaption\>

## \</figure\>

![][image2]

## \<figure\>

## \<img src=’token.png' style='width: 600px'\>

## \<figcaption\>The permissions to give the token.\</figcaption\>

## \</figure\>

* Enable GitHub pages for the repository under *Settings*, if not already enabled.   
* Update the *password* parameter in main.html to a password of your choosing. The default password is **selforgtheworld**.  
* Navigate to *https://username.github.com/post--canonicalized-article-name/public* to see a live draft of the page. This draft typically updates within 30-40 seconds after running an export. The link to the rendered version of this tutorial is [here](https://eyvind.me/gdocs_distill_template/public/)**.**

## (Optional) Adding to Existing Google Doc

Alternatively you can install the script manually on an existing Google Doc.

* Press Tools-\>Script Editor. The new “Apps Script” tab will open.  
* In the Script Editor tab replace the content of the “Code.gs” script with the code from “bin/gdoc2html.gs”. You can also rename “Untitled project” into something meaningful.  
* Go back to the document tab and reload it. The new “HTML Export” menu will appear.

## Features

### Citations

To insert citations, edit the file "*public/bibliography.bib*" and add citations in the BibTex format.  Then, in the text, simply insert a citation as can be seen in the Google Doc \[\[mordvintsev\_niklasson\]\]. This will show up as a footnote with a bibliography at the end, placed using the \<d-citation-list\> section.

### Colab Button

There is built in support for a "Try in Colab" button. To insert this, make sure you have the ***colab*** constant defined in the constants table in the doc. Simple write   
colab(lyxeGm6dJX8D)   
to insert a colab link. This will be rendered as a link with a *scrollTo* the specified section of the colab.

### Hyperlinks

The export script respects hyperlinks and hyperlinked text in Google Docs, and faithfully reproduces it in the rendered HTML.

### Smart Quotes

By default Google Docs uses "smart quotes" (UTF-8 quotes which differ based on whether they are at the beginning or end of a phrase). This does not play well with HTML. The export script replaces these quotes with the standard ASCII double and single quotes.

### Lists and sublists

* The script faithfully reproduces lists as HTML lists.   
  * It respects nesting.  
    * Even discontinuous nesting  
* To change the appearance of the HTML lists, please edit the stylesheet.

### Latex

The underlying Distill template has Latex compilation enabled. To insert equations, simply wrap them as follows: $(\\vec{x} \> \\vec{y}) \\, \\forall \\, \\vec{y}$.

### Custom HTML

Most special characters (for HTML) in the Google Doc are escaped (\<, \>, /, …). However, you can add HTML to have it be exported as-is to the generated HTML in GitHub. To do so, simply mark it as having the *Subtitle* paragraph style in Google Docs. If you duplicated this template, we have adjusted the *Subtitle* paragraph style to be visually unobtrusive. If you are using your own document from scratch, the functionality will still work and you can alter the *Subtitle* paragraph style to look how you want. For an example of inline HTML in the Google Doc, see the below footer code (in the rendered page, you will see the bibliography).

### Videos

There is some simple boilerplate to add videos with an overlaid "play" button in the template. Feel free to customize this, but for a simple video use the following HTML snippet. Customize the "\#t=x.x" to change the preview image for the video (prior to playing) to a specific time in the video, and customize the “src” attribute to choose the video.

![][image3]

## \<figure\>

##   \<div class="vc"\>

##     \<div class="vidoverlay"\>\</div\>

##       \<video playsinline muted width="300px" preload="auto"\>

##         \<source src="grid.mov\#t=0.3" type="video/mp4"\>

##         Your browser does not support the video tag.

##       \</video\>

##   \</div\>

##   \<figcaption\>

## Example of an inline video.

##   \</figcaption\>

## \</figure\>

### Apps Script Changelog

* 16/01/2021  
  * Allow links in footnotes.  
  * Fix URI fragment generation (longer fragments for uniqueness \+ fixed bug) 

## \</d-article\>

## \<d-appendix\>

##     \<d-footnote-list\>\</d-footnote-list\>

##     \<d-citation-list\>\</d-citation-list\>

## \</d-appendix\>