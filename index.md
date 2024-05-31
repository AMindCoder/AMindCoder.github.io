---
layout: default
title: Home
---

# Welcome to My Site

[Introduction to Embedding Models](topics/embedding_models/Introduction_to_Embedding_Models_seo.md)

[Popular Embedding Models](topics/popular_embedding_models/Popular_Embedding_Models_seo.md)

## Topics

<ul>
  {% assign sorted_topics = site.topics | sort: 'path' %}
  {% assign topic_folders = "" | split: "" %}
  {% for topic in sorted_topics %}
    {% assign folder = topic.path | split: "/" | slice: 1, -1 | join: "/" %}
    {% if folder != "" %}
      {% unless topic_folders contains folder %}
        {% assign topic_folders = topic_folders | push: folder %}
      {% endunless %}
    {% endif %}
  {% endfor %}
  
  {% for folder in topic_folders %}
    <li><a href="{{ site.baseurl }}/{{ folder }}">{{ folder }}</a></li>
  {% endfor %}
</ul>