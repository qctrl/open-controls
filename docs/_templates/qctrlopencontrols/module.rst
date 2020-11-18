.. The custom autosummary implementation for Qctrl.

.. _{{fullname}}:

Driven Control
{{ underline }}

.. currentmodule:: {{ module.split('.')[0] }}

.. autosummary::
   :nosignatures:
   :toctree: .

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

