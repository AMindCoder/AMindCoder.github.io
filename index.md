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
  {% if sorted_topics %}
    {% for topic in sorted_topics %}
      <li><a href="{{ topic.url }}">{{ topic.title }}</a></li>
    {% endfor %}
  {% else %}
    <li>No topics found</li>
  {% endif %}
</ul>

## Debugging Output

<p>site.topics: {{ site.topics | jsonify }}</p>
