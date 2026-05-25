// MathJax config covering both rendering paths on this site:
//   - Markdown pages via pymdownx.arithmatex (generic): math arrives as
//     \(...\) / \[...\] inside `arithmatex` spans.
//   - Jupyter notebooks via mkdocs-jupyter: math arrives as raw $...$ / $$...$$.
// So we accept both delimiter styles and let MathJax scan the whole page (it
// skips <pre>/<code>/<script> by default, so $ in code is untouched).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
};

// Re-typeset on Material's instant-navigation page loads.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
