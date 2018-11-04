# Determining the best bandwidth for page-per-visit prediction

See [this blog post](http://dtolpin.github.io/posts/session-depth/) for
details.

* An internet article is split into pages.
* Advertisements are shown on every page.
* The ith visitor ‘churns’ after Kth page.
* We want to forecast the number of pages.

The model is a vector of Beta-Bernoulli distributions.
The bandwidth accounts for changes.
