.. _{{fullname}}:

{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
     .. rubric:: Methods
     .. autosummary::
       :nosignatures:
       :template: autosummary/method.rst
       :toctree: .
       :recursive:
     {% for item in methods %}
     {%- if not item.startswith('_') %}
       ~{{ name }}.{{ item }}
     {%- endif -%}
     {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
     .. rubric:: Attributes
     .. autosummary::
       :nosignatures:
       :template: autosummary/attribute.rst
       :toctree: .
       :recursive:
     {% for item in attributes %}
       ~{{ name }}.{{ item }}
     {%- endfor %}
   {% endif %}
   {% endblock %}
