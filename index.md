---
layout: default
title: Home
---

# Welcome to My Site

[Introduction to Embedding Models](_topics/_embedding_models/Introduction_to_Embedding_Models_seo.md)

## Topics

<ul>
  {% assign sorted_topics = site.topics | sort: 'path' %}
  {% for topic in sorted_topics %}
    <li><a href="{{ topic.url }}">{{ topic.title }}</a></li>
  {% endfor %}
</ul>
