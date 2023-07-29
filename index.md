---
layout: default
title: Index
---

# Index

{% for post in site.posts %}
- [{{ post.name }}]({{ '/posts' | append: post.path | remove_first: '_posts' }})
{% endfor %}

