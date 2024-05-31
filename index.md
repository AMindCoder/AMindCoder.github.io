---
layout: default
title: Eightgen AI Education
---

Explore the wide range of technical resource to learn Generative AI. All the topics are directly or indirectly related to Gen AI application. Happy Learning!!

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

