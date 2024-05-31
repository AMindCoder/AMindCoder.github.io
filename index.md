---
layout: default
title: Home
---

# Welcome to My Site

## Topics

{% assign topics = site.static_files | where: "path", "/topics/" %}
{% for topic in topics %}
  {% assign foldername = topic.path | split: '/' | slice: 2,1 | first %}
  {% assign filename = topic.path | split: '/' | last %}
  {% if forloop.first or foldername != prev_folder %}
    {% unless forloop.first %}
      </ul>
    {% endunless %}
    <h3>{{ foldername }}</h3>
    <ul>
  {% endif %}
  <li><a href="{{ topic.path }}">{{ filename }}</a></li>
  {% assign prev_folder = foldername %}
  {% if forloop.last %}
    </ul>
  {% endif %}
{% endfor %}


