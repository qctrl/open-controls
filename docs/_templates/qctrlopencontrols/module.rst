.. The custom autosummary implementation for Qctrl.

.. _{{fullname}}:

{{ underline }}

.. autosummary::
   :nosignatures:
   :toctree: {{ module.split('.')[0] }}

   {% for item in classes %}
   {%- if not item.startswith('_') %}
       ~{{ module.split('.')[0] }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% for item in functions %}
   {%- if not item.startswith('_') %}
       ~{{ module.split('.')[0] }}.{{ item }}
   {%- endif -%}
   {%- endfor %}

